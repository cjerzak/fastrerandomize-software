#' QJEData: Agricultural Treatment Experiment Data
#'
#' @description
#' Data from a field experiment studying moral hazard in tenancy contracts in agriculture.
#'
#' After subsetting, this dataset includes observations on 968 experimental units
#' with the following variables of interest: household composition,
#' treatment assignment, and agricultural outcomes.
#'
#' @format A data frame with 968 rows and 7 columns:
#' \describe{
#'   \item{children}{Numeric (integer). Number of children in the household. Larger numbers may reflect increased household labor needs and different investment or effort incentives.}
#'   \item{married}{Numeric/binary. Whether the household head is currently married (1) or not (0). Marital status may influence decision-making and risk preferences in farming.}
#'   \item{hh_size}{Numeric (integer). Household size. Differences in family labor availability or consumption needs can influence effort levels and thus relate to moral hazard in production decisions.}
#'   \item{hh_sexrat}{Numeric. The ratio of adult men to adult women in the household. Imbalances in the male–female ratio can affect labor division and investment decisions.}
#'   \item{treat1}{Numeric/binary. Primary treatment indicator (e.g., whether a farmer is offered a specific tenancy contract or cost-sharing arrangement).}
#'   \item{R_yield_ELA_sqm}{Numeric. Crop yield per square meter (e.g., kilograms of output per square meter). This is a principal outcome measure for evaluating productivity and treatment impact on farm performance.}
#'   \item{ELA_Fertil_D}{Numeric/binary. Indicator for whether fertilizer was used (1) or not (0). This measures input investment—a key mechanism in moral hazard models (farmers may alter input use under different contracts).}
#' }
#'
#' @source
#' Burchardi, K.B., Ghatak, M., & Johanssen, A. (2019). 
#' Moral hazard: Experimental evidence from tenancy contracts.
#' \emph{The Quarterly Journal of Economics}, 134(1), 281-347.
#' 
#' @name QJEData
#' @docType data
#' @usage data(QJEData)
NULL

