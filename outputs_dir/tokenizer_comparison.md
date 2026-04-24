# Tokenizer comparison (resnet encoder)

Same encoder and decoder hyperparameters; only tokenizer differs.

## exact_match

| split          |   bert |   custom |
|:---------------|-------:|---------:|
| Numbers_Data   |      0 |   0.0013 |
| Shapes_Data    |      0 |   0      |
| TicTacToe_Data |      0 |   0      |
| overall        |      0 |   0.0004 |

## bleu4

| split          |    bert |   custom |
|:---------------|--------:|---------:|
| Numbers_Data   | 19.4365 |  18.7589 |
| Shapes_Data    | 21.7905 |  25.5944 |
| TicTacToe_Data | 35.6334 |  36.2416 |
| overall        | 26.336  |  28.1416 |

## meteor

| split          |   bert |   custom |
|:---------------|-------:|---------:|
| Numbers_Data   | 0.3511 |   0.3055 |
| Shapes_Data    | 0.4309 |   0.4418 |
| TicTacToe_Data | 0.5378 |   0.498  |
| overall        | 0.4402 |   0.4153 |

## bertscore_f1

| split          |   bert |   custom |
|:---------------|-------:|---------:|
| Numbers_Data   | 0.7326 |   0.7053 |
| Shapes_Data    | 0.749  |   0.7686 |
| TicTacToe_Data | 0.8073 |   0.8084 |
| overall        | 0.7631 |   0.7609 |

