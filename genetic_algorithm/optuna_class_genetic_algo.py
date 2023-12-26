import numpy as np
import optuna
import matplotlib.pyplot as plt
import random

class CustomGeneticAlgorithm(optuna.samplers.BaseSampler):
    def __init__(self, Npop = 100, Ne = 50, Ntournament = 2, pmutQuant = .5, pmutCat = .25, sigma = 1, sigma_halv_thresh = 6):
        self._rng = np.random.RandomState()
        self._current_trial = None  # Current state.
        # add genetic algorithm hyperparam
        # # genetic algo hp
        self.Npop = Npop
        self.Ntournament = Ntournament
        self.Ne = Ne
        # self.NbGen = NbGen
        self.pmutQuant = pmutQuant
        self.pmutCat = pmutCat
        self.sigma = sigma
        self.sigma_halv_thresh = sigma_halv_thresh

    # Keep top Npop best finished trials
    def keepBestNPopTrials(self,study):
      # All completed trials
      listOfCompletedTrials = study.get_trials(states = [optuna.trial.TrialState.COMPLETE])
      # Number of child already computed in the current generation
      NbChildAlreadyCompleted = ((len(listOfCompletedTrials)-self.Npop) % self.Ne)
      # Sample only from population, remove children
      if(NbChildAlreadyCompleted != 0):
          listOfCompletedTrials = listOfCompletedTrials[:-NbChildAlreadyCompleted]
      # Get best individuals from population
      top_Npop = sorted(listOfCompletedTrials, key=lambda x: x.values)[:self.Npop]
      return top_Npop
      
    ## 1 parent selection by tournament
    # ntournament : number of candidate for the tournament
    def tournamentSelection(self,study):
      # Select challengers from completed trials
      listOfCompletedTrials = self.keepBestNPopTrials(study)

      # Initialise storage
      results = [] # initialize an empty list to store the results
      min_value_index = None
      min_value = float('inf')  # initialize to infinity
      
      # select each challenger from the last Npop completed trials
      # store the best challenger (min_value_index)
      for i in range(self.Ntournament):
          indice_i = max(1, round(self.Npop * np.random.rand()))
          value_i = listOfCompletedTrials[-indice_i].value
          results.append((indice_i, value_i))
          if value_i < min_value:
              min_value = value_i
              min_value_index = indice_i
      
      # return the best challenger
      return listOfCompletedTrials[-min_value_index]

    ## 2 parents selection by tournament
    def selection(self,study):
      pere = self.tournamentSelection(study)
      mere = self.tournamentSelection(study)
      return pere, mere

    def crossoverMutation(self, pere, mere, search_space):
      
      params = {}
      
      for param_name, param_distribution in search_space.items():
          
          param_pere = pere.params[param_name]
          param_mere = mere.params[param_name]
          
          param_width = self.sigma_halv_thresh + 1
          # log transform if needed
          if(hasattr(param_distribution, 'log')):
              if(param_distribution.log):
                  param_pere = np.log10(param_pere)
                  param_mere = np.log10(param_mere)
                  param_width = np.log10(param_distribution.high) - np.log10(param_distribution.low)
                    
          # Mother/Father equilibrium
          alp = np.random.rand(1)
          # Mutation
          
          # modifiy sigma if param distribution width is small
          sigma_update = self.sigma
          if(param_width <= self.sigma_halv_thresh):
              sigma_update = self.sigma/10
          
          boolMutationQuant = np.random.rand(1) < self.pmutQuant
          boolMutationCat = np.random.rand(1) < self.pmutCat
          
          if(isinstance(param_distribution, optuna.distributions.CategoricalDistribution)):
              # how to update categorical parameter : either the one from father or mother
              if(alp > 0.5):
                  param_child = param_pere
              else :
                  param_child = param_mere
              if(boolMutationCat):
                  param_child = np.random.choice(param_distribution.choices)
              
          if(isinstance(param_distribution, optuna.distributions.IntDistribution)):
              # how to update integer parameter
              # Barycentric type crossover
              # mutation plus or minus 1
              
              # handle exception seed
              if(param_name == "seed"):
                  if(alp > 0.5):
                      param_child = param_pere
                  else :
                      param_child = param_mere
                  if(boolMutationQuant):
                      param_child = random.randint(param_distribution.low, param_distribution.high)
                  param_child = [param_child]
              else :
                  param_child = np.round(alp * param_pere + (1-alp) * param_mere)
                  if(boolMutationQuant):
                      added = np.round(np.random.rand(1))*2-1
                      param_child = param_child + added
          
          if(isinstance(param_distribution, optuna.distributions.FloatDistribution)):
              # how to update float parameter
              # Barycentric type crossover
              # mutation gaussian
              param_child = alp * param_pere + (1-alp) * param_mere
              if(boolMutationQuant):
                  param_child = param_child + sigma_update * np.random.randn(1)
          
          # log transform back if needed
          if(hasattr(param_distribution, 'log')):
              if(param_distribution.log):
                  param_child = 10**param_child
          
          params[param_name] = param_child[0]
        
      return params

    def sample_relative(self, study, trial, search_space):
        # also return empty if not enought trials yet :
        # len(study.trials) < Npop
        listOfCompletedTrials = study.get_trials(states = [optuna.trial.TrialState.COMPLETE])

        if search_space == {} or len(listOfCompletedTrials) < self.Npop:
            return {}

        # Simulated Annealing algorithm.
        # 1. Select parents by tournament
        pere, mere = self.selection(study)

        # 2. Crossover
        params = self.crossoverMutation(pere = pere, mere = mere, search_space = search_space)
        # 3. Mutation
        
        # 4. Return new children parameters
        return params

    # The rest are unrelated to GA algorithm: boilerplate
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)

# # ### example
# def funsquared(x,y):
#     return x**2 + y**2
#     
# def objective(trial):
#     x = trial.suggest_float("x", -10, 10)
#     y = trial.suggest_int("y", -5, 5)
#     a = trial.suggest_categorical("cat", ["A", "B", "C"])
# 
#     fct_value = funsquared(x,y)
# 
#     if(a == "A"):
#       fct_value = fct_value - 5
# 
#     return fct_value
# 
# sampler = CustomGeneticAlgorithm(Npop = 10)
# study = optuna.create_study(study_name='example-study',
#                             storage = 'sqlite:///example.db',
#                             load_if_exists = True,
#                             sampler=sampler, direction='minimize')
# study.optimize(objective, n_trials=100)
# 
# optuna.visualization.matplotlib.plot_optimization_history(study)
# plt.show()
# study.best_params
