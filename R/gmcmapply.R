#' expand.grid-based multi-core multivariate apply (mcmapply) wrapper for fast implementation of nested looping structures
#'
#' @param mvars a named list of variables to be combined and mapped over. These are equivalent to the layers of a nested for loop.
#' @param FUN user-defined function to apply over variables in mvars. N.B. This function is written to use the names of mvars as formal arguments to FUN. Thus, the majority of FUNs need not require any arguments to be specified. As long as the variable object names in FUN match the names of mvars,  gmapply will handle the translation of names(mvars) to FUN.
#' @param SIMPLIFY defaults to TRUE, which appends function output to a tibble of the expand.grid-ed mvars. FALSE exports a list, where length(list) = nrow(expand.grid(mvars)). TODO: implement an abstracted function that converts the list to a named, nested list when SIMPLIFY = FALSE.
#' @param mc.cores number of cores to utilize if running in parallel. Defaults to 1, which implements mapply.
#' @param ... Additional arguments passed to `FUN`.
#'
#' @return compiled returns of FUN
#'
#' @author Nate Hall
#'
#' @examples
#'
#' \dontrun{
#'      # Example 1:
#'      # just make sure variables used in your function appear as the names of mvars
#'      myfunc <- function(...){
#'        return_me <- paste(l3, l1^2 + l2, sep = "_")
#'        return(return_me)
#'      }
#'
#'      mvars <- list(l1 = 1:3,
#'                    l2 = 1:5,
#'                    l3 = letters[1:3])
#'
#'
#'      ### list output (mapply)
#'      outs_gmc <- gmcmapply(mvars, myfunc, SIMPLIFY = FALSE)
#'
#'      ## N.B. This is equivalent to running:
#'      outs <- vector(mode = "list", length = nrow(expand.grid(mvars)))
#'      step <- 1
#'      for(l1 in 1:10){
#'        for(l2 in 1:5){
#'          for(l3 in letters[1:3]){
#'            outs[[step]] <- myfunc(l1,l2,l3)
#'            step <- step +1
#'          }
#'        }
#'      }
#'
#'      # identical(outs_gmc,outs) #TRUE
#'
#'      ### tibble output with returned value of FUN stored in fun_out. This is the tidy-preferred option.
#'      outs <- gmcmapply(mvars, myfunc, SIMPLIFY = TRUE)
#'
#'
#'      ### tibble output run on 3 cores.
#'      outs <- gmcmapply(mvars, myfunc, SIMPLIFY = TRUE, mc.cores = 3)
#'
#'     Example 2. Pass non-default args to FUN.
#'     ## Since the apply functions dont accept full calls as inputs (calls are internal), user can pass arguments to FUN through dots, which can overwrite a default option for FUN.
#      # e.g. apply(x,1,FUN) works and apply(x,1,FUN(arg_to_change= not_default)) does not, the correct way to specify non-default/additional args to FUN is:
#      # gmcmapply(mvars, FUN, arg_to_change = not_default)
#'
#'     ## update myfunc to have a default argument
#'      myfunc <- function(rep_letters = 3, ...){
#'        return_me <- paste(rep(l3, rep_letters), l1^2 + l2, sep = "_")
#'        return(return_me)
#'      }
#'
#'      outs_default <- gmcmapply(mvars, myfunc)
#'      outs_not_default <- gmcmapply(mvars, myfunc, rep_letters = 1)
#'
#' }
#'
#' @importFrom parallel mcmapply
#'
#' @export
gmcmapply <- function(mvars, FUN, SIMPLIFY = TRUE, mc.cores = 1, ...){

  FUN <- match.fun(FUN)
  funArgs <- formals(FUN)[which(names(formals(FUN)) != "...")] # allow for default args to carry over from FUN.

  expand.dots <- list(...) # allows for expanded dot args to be passed as formal args to the user specified function

  # Implement non-default arg substitutions passed through dots.
  if(any(names(funArgs) %in% names(expand.dots))){
    dot_overwrite <- names(funArgs[which(names(funArgs) %in% names(expand.dots))])
    funArgs[dot_overwrite] <- expand.dots[dot_overwrite]

    #for arg naming and matching below.
    expand.dots[dot_overwrite] <- NULL
  }

  char_vals <- names(mvars[sapply(mvars, is.character)]) # For args that are passed as characters, they should be represented as such when being used in FUN.

  ## build grid of mvars to loop over, this ensures that each combination of various inputs is evaluated (equivalent to creating a structure of nested for loops)
  grid <- expand.grid(mvars,KEEP.OUT.ATTRS = FALSE) %>% arrange_all() %>% # arrange from left (captures nested loop ordering accurately). I got rid of of the ex.gr arg stringsAsFactors = FALSE in order to conserve the user's desired ordering.
    mutate_at(char_vals, as.character) # convert back to character

  # specify formals of the function to be evaluated  by merging the grid to mapply over with expanded dot args
  argdefs <- rep(list(bquote()), ncol(grid) + length(expand.dots) + length(funArgs) + 1)
  names(argdefs) <- c(colnames(grid), names(funArgs), names(expand.dots), "...")

  argdefs[which(names(argdefs) %in% names(funArgs))] <- funArgs # replace with proper dot arg inputs.
  argdefs[which(names(argdefs) %in% names(expand.dots))] <- expand.dots # replace with proper dot arg inputs.

  formals(FUN) <- argdefs

  if(SIMPLIFY) {
    grid$fun_out  <- do.call(mcmapply, c(FUN, c(unname(grid), mc.cores = mc.cores))) # mc.cores = 1 == mapply
    complete <- grid %>% tibble()
  } else{
    complete <- do.call(mcmapply, c(FUN, c(unname(grid), SIMPLIFY = FALSE, mc.cores = mc.cores)))
  }
  return(complete)
}
