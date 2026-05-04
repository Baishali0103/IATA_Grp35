# Error analysis summary

## shallow

| dataset        |   exact_match |   colour_error |   size_error |   shape_error |   position_error |   length_error |   xo_content_error |   n |
|:---------------|--------------:|---------------:|-------------:|--------------:|-----------------:|---------------:|-------------------:|----:|
| Numbers_Data   |             0 |          46.77 |        97.72 |          4.3  |            28.9  |          20.03 |                0   | 744 |
| Shapes_Data    |             0 |          40.59 |        99.19 |         85.48 |            92.88 |           7.93 |                0   | 744 |
| TicTacToe_Data |             0 |          38.4  |         0    |          0    |            89.33 |           0    |               85.2 | 750 |

## deep

| dataset        |   exact_match |   colour_error |   size_error |   shape_error |   position_error |   length_error |   xo_content_error |   n |
|:---------------|--------------:|---------------:|-------------:|--------------:|-----------------:|---------------:|-------------------:|----:|
| Numbers_Data   |          0    |          46.64 |        95.83 |          1.21 |            35.75 |          18.95 |                0   | 744 |
| Shapes_Data    |          0    |          44.89 |        98.52 |         84.14 |            89.25 |           2.82 |                0   | 744 |
| TicTacToe_Data |          0.27 |          36.67 |         0    |          0    |            86.93 |           0    |               78.4 | 750 |

## resnet

| dataset        |   exact_match |   colour_error |   size_error |   shape_error |   position_error |   length_error |   xo_content_error |   n |
|:---------------|--------------:|---------------:|-------------:|--------------:|-----------------:|---------------:|-------------------:|----:|
| Numbers_Data   |          0.13 |          42.88 |        94.35 |          4.17 |            36.69 |          15.99 |               0    | 744 |
| Shapes_Data    |          0    |          25.94 |        98.25 |         58.87 |            86.83 |           3.36 |               0    | 744 |
| TicTacToe_Data |          0    |          38.4  |         0    |          0    |            87.6  |           0    |              79.47 | 750 |

