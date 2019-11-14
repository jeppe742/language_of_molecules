drop view  if exists results;
create view results as select
                             case
                                 when target=prediction then 1
                                 else 0
                             end as accuracy,
                             case 
                                when target=prediction then 1 
                                when (target=0 
                                        or target=4
                                        or target=7
                                        or target=8
                                        or target=9) 
                                    and (prediction=0 
                                        or prediction=4
                                        or prediction=7
                                        or prediction=8
                                        or prediction=9) then 1
                                when (target=2 or target=6) and (prediction=2 or target=6) then 1
                                when (target=3 or target=5) and (prediction=3 or prediction=5) then 1
                              
                                else 0
                           end as accuracy_octet,

                    case 
                        when (target=0 
                                        or target=4
                                        or target=7
                                        or target=8
                                        or target=9) 
                                    and (prediction=0 
                                        or prediction=4
                                        or prediction=7
                                        or prediction=8
                                        or prediction=9) then target
                                when (target=2 or target=6) and (prediction=2 or target=6) then target
                                when (target=3 or target=5) and (prediction=3 or prediction=5) then target
                        
                        
                        else prediction
                    end as prediction_octet,
                    prediction,
                    target, cross_entropy,num_masks, predictions.model_id, model_name_short, length 
                          from predictions, models, molecules
                          where predictions.model_id=models.model_id 
                          and predictions.molecule_id=molecules.molecule_id

;