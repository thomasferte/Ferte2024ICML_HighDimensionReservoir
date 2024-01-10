#' FctCleanFeaturesName
#'
#' @description Clean features names (more explicit names, convert French to English)
#'
#' @param vecFeatures A character vector of features names
#'
#' @return A character vector with the cleaned features names.
#' @export
#' @importFrom stringr str_sub
FctCleanFeaturesName <- function(vecFeatures){
  result <- vecFeatures %>%
    gsub(pattern = "_bin", replacement = "") %>%
    gsub(pattern = "_rol2Deriv7", replacement = " (2nd d)") %>%
    gsub(pattern = "_rolDeriv7", replacement = " (1st d)") %>%
    gsub(pattern = "PCR_INTRA_EXTRA_prelevements_intra_hospitaliers_COUNT", replacement = "CHU positive RT-PCR") %>%
    gsub(pattern = "AH.mean", replacement = "Humidity") %>%
    gsub(pattern = "dewpoint.mean", replacement = "Dew point") %>%
    gsub(pattern = "FRACP_", replacement = "% positive RT-PCR ") %>%
    gsub(pattern = "GRP_GIRONDE_", replacement = "Gironde ") %>%
    gsub(pattern = "hosp", replacement = "Hospitalisation") %>%
    gsub(pattern = "IN_HOSP_in_COUNT", replacement = "New hospitalisation") %>%
    gsub(pattern = "IN_ICU_in", replacement = "New ICU") %>%
    gsub(pattern = "TDM_tdm", replacement = "CT scan") %>%
    gsub(pattern = "IPTCC.mean", replacement = "IPTCC") %>%
    gsub(pattern = "ws.mean", replacement = "Wind speed") %>%
    gsub(pattern = "60_90_PLUS_ANS", replacement = "60+ yo") %>%
    gsub(pattern = "0_19_ANS", replacement = "0-19 yo") %>%
    gsub(pattern = "20_59_ANS", replacement = "20-59 yo") %>%
    gsub(pattern = "TOUS_AGES", replacement = "All age groups") %>%
    gsub(pattern = "TESTED_", replacement = "RT-PCR ") %>%
    gsub(pattern = "^P_", replacement = "positive RT-PCR ") %>%
    gsub(pattern = "Gironde P_", replacement = "Gironde positive RT-PCR ") %>%
    gsub(pattern = "SAMU_", replacement = "SAMU ") %>%
    gsub(pattern = "URG_PEL_", replacement = "Emergency (site 1) ") %>%
    gsub(pattern = "URG_SA_", replacement = "Emergency (site 2) ") %>%
    gsub(pattern = "URG_PED_", replacement = "Emergency (pediatric) ") %>%
    gsub(pattern = "URG_", replacement = "Emergency (all site) ") %>%
    gsub(pattern = "WEEKDAY_", replacement = "Weekday : ") %>%
    gsub(pattern = "Vaccin_1dose", replacement = "Vaccine 1 dose") %>%
    gsub(pattern = "cephalee_2", replacement = '"headache"') %>%
    gsub(pattern = "cephalee", replacement = '"headache"') %>%
    gsub(pattern = "fievre", replacement = '"fever"') %>%
    gsub(pattern = "hyperthermie", replacement = '"hyperthermia"') %>%
    gsub(pattern = "dyspnee", replacement = '"dyspnea"') %>%
    gsub(pattern = "anosmie", replacement = '"anosmia"') %>%
    gsub(pattern = "agueusie", replacement = '"Ageusia"') %>%
    gsub(pattern = "covid_19", replacement = '"covid-19"') %>%
    gsub(pattern = "diarrhee", replacement = '"diarrhea"') %>%
    gsub(pattern = "symptomes_lies_au_covid", replacement = '"Covid symptoms"') %>%
    gsub(pattern = "tous_les_termes", replacement = 'All terms') %>%
    gsub(pattern = "toux", replacement = 'cough') %>%
    gsub(pattern = "_COUNT", replacement = "") %>%
    gsub(pattern = "_PERCENT", replacement = " % of sojourn") %>%
    gsub(pattern = "3$", replacement = "3 days]") %>%
    gsub(pattern = "7$", replacement = "7 days]") %>%
    gsub(pattern = "10$", replacement = "10 days]") %>%
    gsub(pattern = "14$", replacement = "14 days]") %>%
    gsub(pattern = "CHU", replacement = "UHB") %>%
    gsub(pattern = "t.mean", replacement = "Temperature") %>%
    gsub(pattern = "RH.mean", replacement = "Relative humidity") %>%
    gsub(pattern = "AH.mean", replacement = "Absolute humidity") %>%
    gsub(pattern = "precip", replacement = "Precipitation") %>%
    gsub(pattern = "\\] \\[", replacement = ", ") %>%
    gsub(pattern = "GIRONDE_HOSP", replacement = "Gironde Hosp") %>%
    gsub(pattern = "Majority_variant_", replacement = "Majority variant : ") %>%
    gsub(pattern = "OUT_HOSP_out", replacement = "Hospitalization discharge") %>%
    gsub(pattern = "OUT_ICU_out", replacement = "ICU discharge") %>%
    gsub(pattern = "PCR_INTRA_EXTRA_prelevements_extra_Hospitalisationitaliers", replacement = "extra-UHB positive RT-PCR") %>%
  trimws()
  return(result)
}
