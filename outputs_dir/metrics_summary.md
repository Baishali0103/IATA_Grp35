# Evaluation summary

| approach    | split          |    n |   exact_match |   bleu4 |   meteor |   bertscore_p |   bertscore_r |   bertscore_f1 |
|:------------|:---------------|-----:|--------------:|--------:|---------:|--------------:|--------------:|---------------:|
| resnet_bert | overall        | 2238 |             0 | 26.336  | 0.440175 |      0.757096 |      0.771131 |       0.763076 |
| resnet_bert | Shapes_Data    |  744 |             0 | 21.7905 | 0.430857 |      0.740251 |      0.760008 |       0.748959 |
| resnet_bert | Numbers_Data   |  744 |             0 | 19.4365 | 0.351083 |      0.718182 |      0.750437 |       0.732627 |
| resnet_bert | TicTacToe_Data |  750 |             0 | 35.6334 | 0.537799 |      0.812409 |      0.802694 |       0.807285 |