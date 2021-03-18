# -*- coding: utf-8 -*-

import numpy as np


COPPER = np.array([
    [0, 0, 0, 255],
    [0, 0, 1, 255],
    [0, 1, 2, 255],
    [1, 2, 3, 255],
    [1, 3, 4, 255],
    [2, 3, 6, 255],
    [2, 4, 7, 255],
    [3, 5, 8, 255],
    [3, 6, 9, 255],
    [4, 7, 11, 255],
    [4, 7, 12, 255],
    [5, 8, 13, 255],
    [5, 9, 14, 255],
    [6, 10, 16, 255],
    [6, 10, 17, 255],
    [7, 11, 18, 255],
    [7, 12, 19, 255],
    [8, 13, 21, 255],
    [8, 14, 22, 255],
    [9, 14, 23, 255],
    [9, 15, 24, 255],
    [10, 16, 26, 255],
    [10, 17, 27, 255],
    [11, 18, 28, 255],
    [11, 18, 29, 255],
    [12, 19, 31, 255],
    [12, 20, 32, 255],
    [13, 21, 33, 255],
    [13, 21, 34, 255],
    [14, 22, 35, 255],
    [14, 23, 37, 255],
    [15, 24, 38, 255],
    [15, 25, 39, 255],
    [16, 25, 40, 255],
    [16, 26, 42, 255],
    [17, 27, 43, 255],
    [17, 28, 44, 255],
    [18, 29, 45, 255],
    [18, 29, 47, 255],
    [19, 30, 48, 255],
    [19, 31, 49, 255],
    [20, 32, 50, 255],
    [20, 32, 52, 255],
    [21, 33, 53, 255],
    [21, 34, 54, 255],
    [22, 35, 55, 255],
    [22, 36, 57, 255],
    [23, 36, 58, 255],
    [23, 37, 59, 255],
    [24, 38, 60, 255],
    [24, 39, 62, 255],
    [25, 39, 63, 255],
    [25, 40, 64, 255],
    [26, 41, 65, 255],
    [26, 42, 66, 255],
    [27, 43, 68, 255],
    [27, 43, 69, 255],
    [28, 44, 70, 255],
    [28, 45, 71, 255],
    [29, 46, 73, 255],
    [29, 47, 74, 255],
    [30, 47, 75, 255],
    [30, 48, 76, 255],
    [31, 49, 78, 255],
    [31, 50, 79, 255],
    [32, 50, 80, 255],
    [32, 51, 81, 255],
    [33, 52, 83, 255],
    [33, 53, 84, 255],
    [34, 54, 85, 255],
    [34, 54, 86, 255],
    [35, 55, 88, 255],
    [35, 56, 89, 255],
    [36, 57, 90, 255],
    [36, 58, 91, 255],
    [37, 58, 93, 255],
    [37, 59, 94, 255],
    [38, 60, 95, 255],
    [38, 61, 96, 255],
    [39, 61, 97, 255],
    [39, 62, 99, 255],
    [40, 63, 100, 255],
    [40, 64, 101, 255],
    [41, 65, 102, 255],
    [41, 65, 104, 255],
    [42, 66, 105, 255],
    [42, 67, 106, 255],
    [43, 68, 107, 255],
    [43, 69, 109, 255],
    [44, 69, 110, 255],
    [44, 70, 111, 255],
    [45, 71, 112, 255],
    [45, 72, 114, 255],
    [46, 72, 115, 255],
    [46, 73, 116, 255],
    [47, 74, 117, 255],
    [47, 75, 119, 255],
    [48, 76, 120, 255],
    [48, 76, 121, 255],
    [49, 77, 122, 255],
    [49, 78, 124, 255],
    [50, 79, 125, 255],
    [50, 79, 126, 255],
    [51, 80, 127, 255],
    [51, 81, 128, 255],
    [52, 82, 130, 255],
    [52, 83, 131, 255],
    [53, 83, 132, 255],
    [53, 84, 133, 255],
    [54, 85, 135, 255],
    [54, 86, 136, 255],
    [55, 87, 137, 255],
    [55, 87, 138, 255],
    [56, 88, 140, 255],
    [56, 89, 141, 255],
    [57, 90, 142, 255],
    [57, 90, 143, 255],
    [58, 91, 145, 255],
    [58, 92, 146, 255],
    [59, 93, 147, 255],
    [59, 94, 148, 255],
    [60, 94, 150, 255],
    [60, 95, 151, 255],
    [61, 96, 152, 255],
    [61, 97, 153, 255],
    [62, 98, 155, 255],
    [62, 98, 156, 255],
    [63, 99, 157, 255],
    [63, 100, 158, 255],
    [64, 101, 159, 255],
    [64, 101, 161, 255],
    [65, 102, 162, 255],
    [65, 103, 163, 255],
    [66, 104, 164, 255],
    [66, 105, 166, 255],
    [67, 105, 167, 255],
    [67, 106, 168, 255],
    [68, 107, 169, 255],
    [68, 108, 171, 255],
    [69, 109, 172, 255],
    [69, 109, 173, 255],
    [70, 110, 174, 255],
    [70, 111, 176, 255],
    [71, 112, 177, 255],
    [71, 112, 178, 255],
    [72, 113, 179, 255],
    [72, 114, 181, 255],
    [73, 115, 182, 255],
    [73, 116, 183, 255],
    [74, 116, 184, 255],
    [74, 117, 186, 255],
    [75, 118, 187, 255],
    [75, 119, 188, 255],
    [76, 119, 189, 255],
    [76, 120, 190, 255],
    [77, 121, 192, 255],
    [77, 122, 193, 255],
    [78, 123, 194, 255],
    [78, 123, 195, 255],
    [79, 124, 197, 255],
    [79, 125, 198, 255],
    [80, 126, 199, 255],
    [80, 127, 200, 255],
    [81, 127, 202, 255],
    [81, 128, 203, 255],
    [82, 129, 204, 255],
    [82, 130, 205, 255],
    [83, 130, 207, 255],
    [83, 131, 208, 255],
    [84, 132, 209, 255],
    [84, 133, 210, 255],
    [85, 134, 212, 255],
    [85, 134, 213, 255],
    [86, 135, 214, 255],
    [86, 136, 215, 255],
    [87, 137, 217, 255],
    [87, 138, 218, 255],
    [88, 138, 219, 255],
    [88, 139, 220, 255],
    [89, 140, 221, 255],
    [89, 141, 223, 255],
    [90, 141, 224, 255],
    [90, 142, 225, 255],
    [91, 143, 226, 255],
    [91, 144, 228, 255],
    [92, 145, 229, 255],
    [92, 145, 230, 255],
    [93, 146, 231, 255],
    [93, 147, 233, 255],
    [94, 148, 234, 255],
    [94, 149, 235, 255],
    [95, 149, 236, 255],
    [95, 150, 238, 255],
    [96, 151, 239, 255],
    [96, 152, 240, 255],
    [97, 152, 241, 255],
    [97, 153, 243, 255],
    [98, 154, 244, 255],
    [98, 155, 245, 255],
    [99, 156, 246, 255],
    [99, 156, 248, 255],
    [100, 157, 249, 255],
    [100, 158, 250, 255],
    [101, 159, 251, 255],
    [101, 159, 252, 255],
    [102, 160, 254, 255],
    [102, 161, 255, 255],
    [103, 162, 0, 255],
    [103, 163, 0, 255],
    [104, 163, 0, 255],
    [104, 164, 0, 255],
    [105, 165, 0, 255],
    [105, 166, 0, 255],
    [106, 167, 0, 255],
    [106, 167, 0, 255],
    [107, 168, 0, 255],
    [107, 169, 0, 255],
    [108, 170, 0, 255],
    [108, 170, 0, 255],
    [109, 171, 0, 255],
    [109, 172, 0, 255],
    [110, 173, 0, 255],
    [110, 174, 0, 255],
    [111, 174, 0, 255],
    [111, 175, 0, 255],
    [112, 176, 0, 255],
    [112, 177, 0, 255],
    [113, 178, 0, 255],
    [113, 178, 0, 255],
    [114, 179, 0, 255],
    [114, 180, 0, 255],
    [115, 181, 0, 255],
    [115, 181, 0, 255],
    [116, 182, 0, 255],
    [116, 183, 0, 255],
    [117, 184, 0, 255],
    [117, 185, 0, 255],
    [118, 185, 0, 255],
    [118, 186, 0, 255],
    [119, 187, 0, 255],
    [119, 188, 0, 255],
    [120, 189, 0, 255],
    [120, 189, 0, 255],
    [121, 190, 0, 255],
    [121, 191, 0, 255],
    [122, 192, 0, 255],
    [122, 192, 0, 255],
    [123, 193, 0, 255],
    [123, 194, 0, 255],
    [124, 195, 0, 255],
    [124, 196, 0, 255],
    [125, 196, 0, 255],
    [125, 197, 0, 255],
    [126, 198, 0, 255],
    [126, 199, 0, 255],
    [127, 199, 0, 255],
], dtype=np.uint8)


SEISMIC = np.array([
    [76, 0, 0, 255],
    [79, 0, 0, 255],
    [82, 0, 0, 255],
    [85, 0, 0, 255],
    [88, 0, 0, 255],
    [90, 0, 0, 255],
    [93, 0, 0, 255],
    [96, 0, 0, 255],
    [99, 0, 0, 255],
    [102, 0, 0, 255],
    [104, 0, 0, 255],
    [107, 0, 0, 255],
    [110, 0, 0, 255],
    [113, 0, 0, 255],
    [116, 0, 0, 255],
    [118, 0, 0, 255],
    [121, 0, 0, 255],
    [124, 0, 0, 255],
    [127, 0, 0, 255],
    [130, 0, 0, 255],
    [133, 0, 0, 255],
    [135, 0, 0, 255],
    [138, 0, 0, 255],
    [141, 0, 0, 255],
    [144, 0, 0, 255],
    [147, 0, 0, 255],
    [149, 0, 0, 255],
    [152, 0, 0, 255],
    [155, 0, 0, 255],
    [158, 0, 0, 255],
    [161, 0, 0, 255],
    [163, 0, 0, 255],
    [166, 0, 0, 255],
    [169, 0, 0, 255],
    [172, 0, 0, 255],
    [175, 0, 0, 255],
    [177, 0, 0, 255],
    [180, 0, 0, 255],
    [183, 0, 0, 255],
    [186, 0, 0, 255],
    [189, 0, 0, 255],
    [192, 0, 0, 255],
    [194, 0, 0, 255],
    [197, 0, 0, 255],
    [200, 0, 0, 255],
    [203, 0, 0, 255],
    [206, 0, 0, 255],
    [208, 0, 0, 255],
    [211, 0, 0, 255],
    [214, 0, 0, 255],
    [217, 0, 0, 255],
    [220, 0, 0, 255],
    [222, 0, 0, 255],
    [225, 0, 0, 255],
    [228, 0, 0, 255],
    [231, 0, 0, 255],
    [234, 0, 0, 255],
    [237, 0, 0, 255],
    [239, 0, 0, 255],
    [242, 0, 0, 255],
    [245, 0, 0, 255],
    [248, 0, 0, 255],
    [251, 0, 0, 255],
    [253, 0, 0, 255],
    [0, 1, 1, 255],
    [0, 5, 5, 255],
    [0, 9, 9, 255],
    [0, 13, 13, 255],
    [0, 17, 17, 255],
    [0, 21, 21, 255],
    [0, 25, 25, 255],
    [0, 29, 29, 255],
    [0, 33, 33, 255],
    [0, 37, 37, 255],
    [0, 41, 41, 255],
    [0, 45, 45, 255],
    [0, 49, 49, 255],
    [0, 53, 53, 255],
    [0, 57, 57, 255],
    [0, 61, 61, 255],
    [0, 65, 65, 255],
    [0, 69, 69, 255],
    [0, 73, 73, 255],
    [0, 77, 77, 255],
    [0, 81, 81, 255],
    [0, 85, 85, 255],
    [0, 89, 89, 255],
    [0, 93, 93, 255],
    [0, 97, 97, 255],
    [0, 101, 101, 255],
    [0, 105, 105, 255],
    [0, 109, 109, 255],
    [0, 113, 113, 255],
    [0, 117, 117, 255],
    [0, 121, 121, 255],
    [0, 125, 125, 255],
    [0, 129, 129, 255],
    [0, 133, 133, 255],
    [0, 137, 137, 255],
    [0, 141, 141, 255],
    [0, 145, 145, 255],
    [0, 149, 149, 255],
    [0, 153, 153, 255],
    [0, 157, 157, 255],
    [0, 161, 161, 255],
    [0, 165, 165, 255],
    [0, 169, 169, 255],
    [0, 173, 173, 255],
    [0, 177, 177, 255],
    [0, 181, 181, 255],
    [0, 185, 185, 255],
    [0, 189, 189, 255],
    [0, 193, 193, 255],
    [0, 197, 197, 255],
    [0, 201, 201, 255],
    [0, 205, 205, 255],
    [0, 209, 209, 255],
    [0, 213, 213, 255],
    [0, 217, 217, 255],
    [0, 221, 221, 255],
    [0, 225, 225, 255],
    [0, 229, 229, 255],
    [0, 233, 233, 255],
    [0, 237, 237, 255],
    [0, 241, 241, 255],
    [0, 245, 245, 255],
    [0, 249, 249, 255],
    [0, 253, 253, 255],
    [253, 253, 0, 255],
    [249, 249, 0, 255],
    [245, 245, 0, 255],
    [241, 241, 0, 255],
    [237, 237, 0, 255],
    [233, 233, 0, 255],
    [229, 229, 0, 255],
    [225, 225, 0, 255],
    [221, 221, 0, 255],
    [217, 217, 0, 255],
    [213, 213, 0, 255],
    [209, 209, 0, 255],
    [205, 205, 0, 255],
    [201, 201, 0, 255],
    [197, 197, 0, 255],
    [193, 193, 0, 255],
    [189, 189, 0, 255],
    [185, 185, 0, 255],
    [181, 181, 0, 255],
    [177, 177, 0, 255],
    [173, 173, 0, 255],
    [169, 169, 0, 255],
    [165, 165, 0, 255],
    [161, 161, 0, 255],
    [157, 157, 0, 255],
    [153, 153, 0, 255],
    [149, 149, 0, 255],
    [145, 145, 0, 255],
    [141, 141, 0, 255],
    [137, 137, 0, 255],
    [133, 133, 0, 255],
    [129, 129, 0, 255],
    [125, 125, 0, 255],
    [121, 121, 0, 255],
    [117, 117, 0, 255],
    [113, 113, 0, 255],
    [109, 109, 0, 255],
    [105, 105, 0, 255],
    [101, 101, 0, 255],
    [97, 97, 0, 255],
    [93, 93, 0, 255],
    [89, 89, 0, 255],
    [85, 85, 0, 255],
    [81, 81, 0, 255],
    [77, 77, 0, 255],
    [73, 73, 0, 255],
    [69, 69, 0, 255],
    [65, 65, 0, 255],
    [61, 61, 0, 255],
    [57, 57, 0, 255],
    [53, 53, 0, 255],
    [49, 49, 0, 255],
    [45, 45, 0, 255],
    [41, 41, 0, 255],
    [37, 37, 0, 255],
    [33, 33, 0, 255],
    [29, 29, 0, 255],
    [25, 25, 0, 255],
    [21, 21, 0, 255],
    [17, 17, 0, 255],
    [13, 13, 0, 255],
    [9, 9, 0, 255],
    [5, 5, 0, 255],
    [1, 1, 0, 255],
    [0, 0, 254, 255],
    [0, 0, 252, 255],
    [0, 0, 250, 255],
    [0, 0, 248, 255],
    [0, 0, 246, 255],
    [0, 0, 244, 255],
    [0, 0, 242, 255],
    [0, 0, 240, 255],
    [0, 0, 238, 255],
    [0, 0, 236, 255],
    [0, 0, 234, 255],
    [0, 0, 232, 255],
    [0, 0, 230, 255],
    [0, 0, 228, 255],
    [0, 0, 226, 255],
    [0, 0, 224, 255],
    [0, 0, 222, 255],
    [0, 0, 220, 255],
    [0, 0, 218, 255],
    [0, 0, 216, 255],
    [0, 0, 214, 255],
    [0, 0, 212, 255],
    [0, 0, 210, 255],
    [0, 0, 208, 255],
    [0, 0, 206, 255],
    [0, 0, 204, 255],
    [0, 0, 202, 255],
    [0, 0, 200, 255],
    [0, 0, 198, 255],
    [0, 0, 196, 255],
    [0, 0, 194, 255],
    [0, 0, 192, 255],
    [0, 0, 190, 255],
    [0, 0, 188, 255],
    [0, 0, 186, 255],
    [0, 0, 184, 255],
    [0, 0, 182, 255],
    [0, 0, 180, 255],
    [0, 0, 178, 255],
    [0, 0, 176, 255],
    [0, 0, 174, 255],
    [0, 0, 172, 255],
    [0, 0, 170, 255],
    [0, 0, 168, 255],
    [0, 0, 166, 255],
    [0, 0, 164, 255],
    [0, 0, 162, 255],
    [0, 0, 160, 255],
    [0, 0, 158, 255],
    [0, 0, 156, 255],
    [0, 0, 154, 255],
    [0, 0, 152, 255],
    [0, 0, 150, 255],
    [0, 0, 148, 255],
    [0, 0, 146, 255],
    [0, 0, 144, 255],
    [0, 0, 142, 255],
    [0, 0, 140, 255],
    [0, 0, 138, 255],
    [0, 0, 136, 255],
    [0, 0, 134, 255],
    [0, 0, 132, 255],
    [0, 0, 130, 255],
    [0, 0, 128, 255],
], dtype=np.uint8)


COOLWARM = np.array([
    [192, 76, 58, 255],
    [194, 78, 60, 255],
    [196, 79, 61, 255],
    [197, 81, 62, 255],
    [199, 83, 63, 255],
    [200, 85, 64, 255],
    [202, 86, 65, 255],
    [203, 88, 67, 255],
    [205, 90, 68, 255],
    [206, 92, 69, 255],
    [208, 93, 70, 255],
    [209, 95, 71, 255],
    [210, 97, 73, 255],
    [212, 99, 74, 255],
    [213, 100, 75, 255],
    [214, 102, 76, 255],
    [216, 104, 77, 255],
    [217, 105, 79, 255],
    [218, 107, 80, 255],
    [220, 109, 81, 255],
    [221, 110, 82, 255],
    [222, 112, 84, 255],
    [223, 114, 85, 255],
    [225, 115, 86, 255],
    [226, 117, 87, 255],
    [227, 119, 89, 255],
    [228, 120, 90, 255],
    [229, 122, 91, 255],
    [230, 124, 93, 255],
    [231, 125, 94, 255],
    [232, 127, 95, 255],
    [233, 128, 96, 255],
    [234, 130, 98, 255],
    [235, 132, 99, 255],
    [236, 133, 100, 255],
    [237, 135, 102, 255],
    [238, 136, 103, 255],
    [239, 138, 104, 255],
    [240, 139, 106, 255],
    [241, 141, 107, 255],
    [242, 143, 108, 255],
    [242, 144, 110, 255],
    [243, 146, 111, 255],
    [244, 147, 112, 255],
    [245, 149, 114, 255],
    [245, 150, 115, 255],
    [246, 152, 117, 255],
    [247, 153, 118, 255],
    [247, 155, 119, 255],
    [248, 156, 121, 255],
    [249, 157, 122, 255],
    [249, 159, 123, 255],
    [250, 160, 125, 255],
    [250, 162, 126, 255],
    [251, 163, 128, 255],
    [251, 164, 129, 255],
    [252, 166, 130, 255],
    [252, 167, 132, 255],
    [252, 168, 133, 255],
    [253, 170, 134, 255],
    [253, 171, 136, 255],
    [253, 172, 137, 255],
    [254, 174, 139, 255],
    [254, 175, 140, 255],
    [254, 176, 141, 255],
    [254, 177, 143, 255],
    [255, 179, 144, 255],
    [255, 180, 146, 255],
    [255, 181, 147, 255],
    [255, 182, 148, 255],
    [255, 183, 150, 255],
    [255, 185, 151, 255],
    [255, 186, 153, 255],
    [255, 187, 154, 255],
    [255, 188, 155, 255],
    [255, 189, 157, 255],
    [255, 190, 158, 255],
    [255, 191, 159, 255],
    [255, 192, 161, 255],
    [255, 193, 162, 255],
    [255, 194, 164, 255],
    [255, 195, 165, 255],
    [254, 196, 166, 255],
    [254, 197, 168, 255],
    [254, 198, 169, 255],
    [254, 199, 170, 255],
    [253, 200, 172, 255],
    [253, 201, 173, 255],
    [253, 202, 174, 255],
    [252, 203, 176, 255],
    [252, 203, 177, 255],
    [252, 204, 178, 255],
    [251, 205, 180, 255],
    [251, 206, 181, 255],
    [250, 207, 182, 255],
    [250, 207, 184, 255],
    [249, 208, 185, 255],
    [249, 209, 186, 255],
    [248, 209, 187, 255],
    [247, 210, 189, 255],
    [247, 211, 190, 255],
    [246, 211, 191, 255],
    [245, 212, 192, 255],
    [245, 213, 194, 255],
    [244, 213, 195, 255],
    [243, 214, 196, 255],
    [243, 214, 197, 255],
    [242, 215, 199, 255],
    [241, 215, 200, 255],
    [240, 216, 201, 255],
    [239, 216, 202, 255],
    [238, 217, 203, 255],
    [238, 217, 204, 255],
    [237, 218, 206, 255],
    [236, 218, 207, 255],
    [235, 218, 208, 255],
    [234, 219, 209, 255],
    [233, 219, 210, 255],
    [232, 219, 211, 255],
    [231, 220, 212, 255],
    [230, 220, 213, 255],
    [228, 220, 214, 255],
    [227, 220, 215, 255],
    [226, 220, 216, 255],
    [225, 220, 217, 255],
    [224, 221, 218, 255],
    [223, 221, 220, 255],
    [222, 221, 221, 255],
    [220, 221, 222, 255],
    [219, 220, 223, 255],
    [217, 220, 224, 255],
    [216, 219, 225, 255],
    [215, 219, 226, 255],
    [213, 218, 227, 255],
    [212, 218, 228, 255],
    [210, 217, 229, 255],
    [209, 216, 230, 255],
    [207, 216, 231, 255],
    [206, 215, 231, 255],
    [204, 214, 232, 255],
    [203, 214, 233, 255],
    [201, 213, 234, 255],
    [200, 212, 235, 255],
    [198, 212, 236, 255],
    [197, 211, 236, 255],
    [195, 210, 237, 255],
    [194, 209, 238, 255],
    [192, 208, 238, 255],
    [191, 207, 239, 255],
    [189, 207, 240, 255],
    [188, 206, 240, 255],
    [186, 205, 241, 255],
    [185, 204, 242, 255],
    [183, 203, 242, 255],
    [181, 202, 242, 255],
    [180, 201, 243, 255],
    [178, 200, 243, 255],
    [177, 199, 244, 255],
    [175, 198, 244, 255],
    [174, 197, 245, 255],
    [172, 196, 245, 255],
    [170, 195, 245, 255],
    [169, 194, 246, 255],
    [167, 192, 246, 255],
    [166, 191, 246, 255],
    [164, 190, 246, 255],
    [163, 189, 247, 255],
    [161, 188, 247, 255],
    [159, 187, 247, 255],
    [158, 185, 247, 255],
    [156, 184, 247, 255],
    [155, 183, 247, 255],
    [153, 181, 248, 255],
    [151, 180, 248, 255],
    [150, 179, 248, 255],
    [148, 178, 248, 255],
    [147, 176, 248, 255],
    [145, 175, 248, 255],
    [144, 173, 248, 255],
    [142, 172, 247, 255],
    [140, 171, 247, 255],
    [139, 169, 247, 255],
    [137, 168, 247, 255],
    [136, 166, 247, 255],
    [134, 165, 247, 255],
    [133, 163, 247, 255],
    [131, 162, 246, 255],
    [129, 160, 246, 255],
    [128, 159, 246, 255],
    [126, 157, 245, 255],
    [125, 156, 245, 255],
    [123, 154, 245, 255],
    [122, 153, 244, 255],
    [120, 151, 244, 255],
    [119, 149, 243, 255],
    [117, 148, 243, 255],
    [116, 146, 243, 255],
    [114, 144, 242, 255],
    [113, 143, 242, 255],
    [111, 141, 241, 255],
    [110, 139, 241, 255],
    [108, 138, 240, 255],
    [107, 136, 239, 255],
    [105, 134, 239, 255],
    [104, 132, 238, 255],
    [102, 131, 237, 255],
    [101, 129, 237, 255],
    [99, 127, 236, 255],
    [98, 125, 235, 255],
    [96, 123, 235, 255],
    [95, 122, 234, 255],
    [93, 120, 233, 255],
    [92, 118, 232, 255],
    [90, 116, 231, 255],
    [89, 114, 231, 255],
    [88, 112, 230, 255],
    [86, 110, 229, 255],
    [85, 108, 228, 255],
    [83, 106, 227, 255],
    [82, 104, 226, 255],
    [81, 102, 225, 255],
    [79, 101, 224, 255],
    [78, 99, 223, 255],
    [77, 97, 222, 255],
    [75, 95, 221, 255],
    [74, 92, 220, 255],
    [73, 90, 219, 255],
    [71, 88, 218, 255],
    [70, 86, 217, 255],
    [69, 84, 215, 255],
    [67, 82, 214, 255],
    [66, 80, 213, 255],
    [65, 78, 212, 255],
    [64, 75, 211, 255],
    [62, 73, 210, 255],
    [61, 71, 208, 255],
    [60, 68, 207, 255],
    [59, 66, 206, 255],
    [57, 64, 205, 255],
    [56, 61, 203, 255],
    [55, 59, 202, 255],
    [54, 56, 200, 255],
    [53, 53, 199, 255],
    [51, 51, 198, 255],
    [50, 48, 196, 255],
    [49, 45, 195, 255],
    [48, 43, 194, 255],
    [47, 40, 192, 255],
    [46, 35, 191, 255],
    [44, 31, 189, 255],
    [43, 26, 188, 255],
    [42, 22, 186, 255],
    [41, 17, 185, 255],
    [40, 13, 183, 255],
    [39, 8, 182, 255],
    [38, 3, 180, 255],
], dtype=np.uint8)


RAINBOW = np.array([
    [0, 0, 128, 255],
    [255, 3, 125, 255],
    [255, 6, 123, 255],
    [255, 9, 121, 255],
    [255, 12, 119, 255],
    [255, 15, 117, 255],
    [255, 18, 115, 255],
    [255, 22, 113, 255],
    [255, 25, 111, 255],
    [255, 28, 109, 255],
    [255, 31, 107, 255],
    [255, 34, 105, 255],
    [255, 37, 103, 255],
    [255, 40, 101, 255],
    [255, 43, 99, 255],
    [254, 47, 97, 255],
    [254, 50, 95, 255],
    [254, 53, 93, 255],
    [254, 56, 91, 255],
    [254, 59, 89, 255],
    [254, 62, 87, 255],
    [253, 65, 85, 255],
    [253, 68, 83, 255],
    [253, 71, 81, 255],
    [253, 74, 79, 255],
    [252, 77, 77, 255],
    [252, 80, 75, 255],
    [252, 83, 73, 255],
    [252, 86, 71, 255],
    [251, 89, 69, 255],
    [251, 92, 67, 255],
    [251, 95, 65, 255],
    [251, 98, 63, 255],
    [250, 101, 61, 255],
    [250, 104, 59, 255],
    [250, 106, 57, 255],
    [249, 109, 55, 255],
    [249, 112, 53, 255],
    [249, 115, 51, 255],
    [248, 118, 49, 255],
    [248, 121, 47, 255],
    [247, 123, 45, 255],
    [247, 126, 43, 255],
    [247, 129, 41, 255],
    [246, 132, 39, 255],
    [246, 134, 37, 255],
    [245, 137, 35, 255],
    [245, 140, 33, 255],
    [244, 142, 31, 255],
    [244, 145, 29, 255],
    [243, 147, 27, 255],
    [243, 150, 25, 255],
    [242, 153, 23, 255],
    [242, 155, 21, 255],
    [241, 158, 19, 255],
    [241, 160, 17, 255],
    [240, 162, 15, 255],
    [240, 165, 13, 255],
    [239, 167, 11, 255],
    [239, 170, 9, 255],
    [238, 172, 7, 255],
    [238, 174, 5, 255],
    [237, 177, 3, 255],
    [236, 179, 1, 255],
    [236, 181, 0, 255],
    [235, 183, 2, 255],
    [235, 185, 4, 255],
    [234, 188, 6, 255],
    [233, 190, 8, 255],
    [233, 192, 10, 255],
    [232, 194, 12, 255],
    [231, 196, 14, 255],
    [231, 198, 16, 255],
    [230, 200, 18, 255],
    [229, 202, 20, 255],
    [229, 204, 22, 255],
    [228, 206, 24, 255],
    [227, 208, 26, 255],
    [227, 209, 28, 255],
    [226, 211, 30, 255],
    [225, 213, 32, 255],
    [224, 215, 34, 255],
    [224, 216, 36, 255],
    [223, 218, 38, 255],
    [222, 220, 40, 255],
    [221, 221, 42, 255],
    [220, 223, 44, 255],
    [220, 224, 46, 255],
    [219, 226, 48, 255],
    [218, 227, 50, 255],
    [217, 229, 52, 255],
    [216, 230, 54, 255],
    [215, 231, 56, 255],
    [215, 233, 58, 255],
    [214, 234, 60, 255],
    [213, 235, 62, 255],
    [212, 236, 64, 255],
    [211, 238, 66, 255],
    [210, 239, 68, 255],
    [209, 240, 70, 255],
    [208, 241, 72, 255],
    [208, 242, 74, 255],
    [207, 243, 76, 255],
    [206, 244, 78, 255],
    [205, 245, 80, 255],
    [204, 246, 82, 255],
    [203, 247, 84, 255],
    [202, 247, 86, 255],
    [201, 248, 88, 255],
    [200, 249, 90, 255],
    [199, 250, 92, 255],
    [198, 250, 94, 255],
    [197, 251, 96, 255],
    [196, 251, 98, 255],
    [195, 252, 100, 255],
    [194, 252, 102, 255],
    [193, 253, 104, 255],
    [192, 253, 106, 255],
    [191, 254, 108, 255],
    [190, 254, 110, 255],
    [189, 254, 112, 255],
    [188, 255, 114, 255],
    [187, 255, 116, 255],
    [185, 255, 118, 255],
    [184, 255, 120, 255],
    [183, 255, 122, 255],
    [182, 255, 124, 255],
    [181, 255, 126, 255],
    [180, 255, 129, 255],
    [179, 255, 131, 255],
    [178, 255, 133, 255],
    [177, 255, 135, 255],
    [175, 255, 137, 255],
    [174, 255, 139, 255],
    [173, 255, 141, 255],
    [172, 254, 143, 255],
    [171, 254, 145, 255],
    [170, 254, 147, 255],
    [168, 253, 149, 255],
    [167, 253, 151, 255],
    [166, 252, 153, 255],
    [165, 252, 155, 255],
    [164, 251, 157, 255],
    [162, 251, 159, 255],
    [161, 250, 161, 255],
    [160, 250, 163, 255],
    [159, 249, 165, 255],
    [158, 248, 167, 255],
    [156, 247, 169, 255],
    [155, 247, 171, 255],
    [154, 246, 173, 255],
    [153, 245, 175, 255],
    [151, 244, 177, 255],
    [150, 243, 179, 255],
    [149, 242, 181, 255],
    [147, 241, 183, 255],
    [146, 240, 185, 255],
    [145, 239, 187, 255],
    [144, 238, 189, 255],
    [142, 236, 191, 255],
    [141, 235, 193, 255],
    [140, 234, 195, 255],
    [138, 233, 197, 255],
    [137, 231, 199, 255],
    [136, 230, 201, 255],
    [134, 229, 203, 255],
    [133, 227, 205, 255],
    [132, 226, 207, 255],
    [130, 224, 209, 255],
    [129, 223, 211, 255],
    [128, 221, 213, 255],
    [126, 220, 215, 255],
    [125, 218, 217, 255],
    [123, 216, 219, 255],
    [122, 215, 221, 255],
    [121, 213, 223, 255],
    [119, 211, 225, 255],
    [118, 209, 227, 255],
    [116, 208, 229, 255],
    [115, 206, 231, 255],
    [114, 204, 233, 255],
    [112, 202, 235, 255],
    [111, 200, 237, 255],
    [109, 198, 239, 255],
    [108, 196, 241, 255],
    [106, 194, 243, 255],
    [105, 192, 245, 255],
    [104, 190, 247, 255],
    [102, 188, 249, 255],
    [101, 185, 251, 255],
    [99, 183, 253, 255],
    [98, 181, 255, 255],
    [96, 179, 0, 255],
    [95, 177, 0, 255],
    [93, 174, 0, 255],
    [92, 172, 0, 255],
    [91, 170, 0, 255],
    [89, 167, 0, 255],
    [88, 165, 0, 255],
    [86, 162, 0, 255],
    [85, 160, 0, 255],
    [83, 158, 0, 255],
    [82, 155, 0, 255],
    [80, 153, 0, 255],
    [79, 150, 0, 255],
    [77, 147, 0, 255],
    [76, 145, 0, 255],
    [74, 142, 0, 255],
    [73, 140, 0, 255],
    [71, 137, 0, 255],
    [70, 134, 0, 255],
    [68, 132, 0, 255],
    [67, 129, 0, 255],
    [65, 126, 0, 255],
    [63, 123, 0, 255],
    [62, 121, 0, 255],
    [60, 118, 0, 255],
    [59, 115, 0, 255],
    [57, 112, 0, 255],
    [56, 109, 0, 255],
    [54, 106, 0, 255],
    [53, 104, 0, 255],
    [51, 101, 0, 255],
    [50, 98, 0, 255],
    [48, 95, 0, 255],
    [47, 92, 0, 255],
    [45, 89, 0, 255],
    [43, 86, 0, 255],
    [42, 83, 0, 255],
    [40, 80, 0, 255],
    [39, 77, 0, 255],
    [37, 74, 0, 255],
    [36, 71, 0, 255],
    [34, 68, 0, 255],
    [33, 65, 0, 255],
    [31, 62, 0, 255],
    [29, 59, 0, 255],
    [28, 56, 0, 255],
    [26, 53, 0, 255],
    [25, 50, 0, 255],
    [23, 47, 0, 255],
    [22, 43, 0, 255],
    [20, 40, 0, 255],
    [18, 37, 0, 255],
    [17, 34, 0, 255],
    [15, 31, 0, 255],
    [14, 28, 0, 255],
    [12, 25, 0, 255],
    [11, 22, 0, 255],
    [9, 18, 0, 255],
    [7, 15, 0, 255],
    [6, 12, 0, 255],
    [4, 9, 0, 255],
    [3, 6, 0, 255],
    [1, 3, 0, 255],
    [0, 0, 0, 255],
], dtype=np.uint8)


YLGN = np.array([
    [229, 0, 0, 255],
    [228, 255, 255, 255],
    [227, 255, 255, 255],
    [225, 255, 255, 255],
    [224, 255, 254, 255],
    [222, 255, 254, 255],
    [221, 255, 254, 255],
    [220, 255, 254, 255],
    [218, 255, 253, 255],
    [217, 255, 253, 255],
    [216, 255, 253, 255],
    [214, 254, 253, 255],
    [213, 254, 252, 255],
    [211, 254, 252, 255],
    [210, 254, 252, 255],
    [209, 254, 252, 255],
    [207, 254, 251, 255],
    [206, 254, 251, 255],
    [204, 254, 251, 255],
    [203, 254, 251, 255],
    [202, 254, 250, 255],
    [200, 254, 250, 255],
    [199, 253, 250, 255],
    [198, 253, 250, 255],
    [196, 253, 249, 255],
    [195, 253, 249, 255],
    [193, 253, 249, 255],
    [192, 253, 249, 255],
    [191, 253, 248, 255],
    [189, 253, 248, 255],
    [188, 253, 248, 255],
    [186, 253, 248, 255],
    [185, 252, 247, 255],
    [184, 252, 246, 255],
    [184, 252, 245, 255],
    [183, 251, 245, 255],
    [182, 251, 244, 255],
    [182, 251, 243, 255],
    [181, 250, 242, 255],
    [180, 250, 241, 255],
    [180, 249, 240, 255],
    [179, 249, 239, 255],
    [178, 249, 238, 255],
    [178, 248, 237, 255],
    [177, 248, 236, 255],
    [176, 248, 235, 255],
    [175, 247, 234, 255],
    [175, 247, 233, 255],
    [174, 246, 232, 255],
    [173, 246, 231, 255],
    [173, 246, 230, 255],
    [172, 245, 229, 255],
    [171, 245, 228, 255],
    [171, 245, 228, 255],
    [170, 244, 227, 255],
    [169, 244, 226, 255],
    [169, 243, 225, 255],
    [168, 243, 224, 255],
    [167, 243, 223, 255],
    [166, 242, 222, 255],
    [166, 242, 221, 255],
    [165, 241, 220, 255],
    [164, 241, 219, 255],
    [164, 241, 218, 255],
    [163, 240, 217, 255],
    [162, 240, 216, 255],
    [162, 239, 214, 255],
    [161, 238, 213, 255],
    [160, 238, 211, 255],
    [160, 237, 210, 255],
    [159, 237, 209, 255],
    [158, 236, 207, 255],
    [158, 236, 206, 255],
    [157, 235, 205, 255],
    [156, 234, 203, 255],
    [156, 234, 202, 255],
    [155, 233, 200, 255],
    [154, 233, 199, 255],
    [154, 232, 198, 255],
    [153, 231, 196, 255],
    [152, 231, 195, 255],
    [152, 230, 193, 255],
    [151, 230, 192, 255],
    [150, 229, 191, 255],
    [150, 228, 189, 255],
    [149, 228, 188, 255],
    [148, 227, 187, 255],
    [148, 227, 185, 255],
    [147, 226, 184, 255],
    [146, 225, 182, 255],
    [146, 225, 181, 255],
    [145, 224, 180, 255],
    [144, 224, 178, 255],
    [144, 223, 177, 255],
    [143, 222, 175, 255],
    [142, 222, 174, 255],
    [142, 221, 173, 255],
    [141, 220, 171, 255],
    [140, 220, 169, 255],
    [140, 219, 168, 255],
    [139, 218, 166, 255],
    [139, 217, 164, 255],
    [138, 217, 163, 255],
    [137, 216, 161, 255],
    [137, 215, 159, 255],
    [136, 215, 158, 255],
    [135, 214, 156, 255],
    [135, 213, 154, 255],
    [134, 212, 153, 255],
    [133, 212, 151, 255],
    [133, 211, 149, 255],
    [132, 210, 148, 255],
    [131, 210, 146, 255],
    [131, 209, 144, 255],
    [130, 208, 143, 255],
    [129, 207, 141, 255],
    [129, 207, 139, 255],
    [128, 206, 137, 255],
    [127, 205, 136, 255],
    [127, 204, 134, 255],
    [126, 204, 132, 255],
    [125, 203, 131, 255],
    [125, 202, 129, 255],
    [124, 202, 127, 255],
    [123, 201, 126, 255],
    [123, 200, 124, 255],
    [122, 199, 122, 255],
    [121, 199, 121, 255],
    [121, 198, 119, 255],
    [120, 197, 117, 255],
    [119, 196, 116, 255],
    [118, 195, 114, 255],
    [117, 194, 112, 255],
    [116, 194, 110, 255],
    [115, 193, 109, 255],
    [114, 192, 107, 255],
    [113, 191, 105, 255],
    [113, 190, 104, 255],
    [112, 189, 102, 255],
    [111, 188, 100, 255],
    [110, 188, 98, 255],
    [109, 187, 97, 255],
    [108, 186, 95, 255],
    [107, 185, 93, 255],
    [106, 184, 91, 255],
    [106, 183, 90, 255],
    [105, 183, 88, 255],
    [104, 182, 86, 255],
    [103, 181, 84, 255],
    [102, 180, 83, 255],
    [101, 179, 81, 255],
    [100, 178, 79, 255],
    [99, 177, 78, 255],
    [98, 177, 76, 255],
    [98, 176, 74, 255],
    [97, 175, 72, 255],
    [96, 174, 71, 255],
    [95, 173, 69, 255],
    [94, 172, 67, 255],
    [93, 171, 65, 255],
    [92, 170, 64, 255],
    [92, 169, 63, 255],
    [91, 168, 62, 255],
    [90, 167, 61, 255],
    [89, 165, 60, 255],
    [88, 164, 59, 255],
    [87, 163, 58, 255],
    [87, 162, 58, 255],
    [86, 161, 57, 255],
    [85, 159, 56, 255],
    [84, 158, 55, 255],
    [83, 157, 54, 255],
    [83, 156, 53, 255],
    [82, 154, 52, 255],
    [81, 153, 51, 255],
    [80, 152, 50, 255],
    [79, 151, 49, 255],
    [78, 150, 48, 255],
    [78, 148, 47, 255],
    [77, 147, 46, 255],
    [76, 146, 45, 255],
    [75, 145, 44, 255],
    [74, 143, 43, 255],
    [74, 142, 42, 255],
    [73, 141, 41, 255],
    [72, 140, 41, 255],
    [71, 138, 40, 255],
    [70, 137, 39, 255],
    [69, 136, 38, 255],
    [69, 135, 37, 255],
    [68, 134, 36, 255],
    [67, 132, 35, 255],
    [66, 131, 34, 255],
    [66, 130, 33, 255],
    [66, 130, 32, 255],
    [65, 129, 31, 255],
    [65, 128, 29, 255],
    [65, 127, 28, 255],
    [64, 126, 27, 255],
    [64, 125, 26, 255],
    [63, 124, 25, 255],
    [63, 123, 24, 255],
    [63, 123, 23, 255],
    [62, 122, 22, 255],
    [62, 121, 21, 255],
    [62, 120, 19, 255],
    [61, 119, 18, 255],
    [61, 118, 17, 255],
    [60, 117, 16, 255],
    [60, 116, 15, 255],
    [60, 115, 14, 255],
    [59, 115, 13, 255],
    [59, 114, 12, 255],
    [59, 113, 11, 255],
    [58, 112, 10, 255],
    [58, 111, 8, 255],
    [57, 110, 7, 255],
    [57, 109, 6, 255],
    [57, 108, 5, 255],
    [56, 108, 4, 255],
    [56, 107, 3, 255],
    [56, 106, 2, 255],
    [55, 105, 1, 255],
    [55, 104, 0, 255],
    [54, 103, 0, 255],
    [54, 102, 0, 255],
    [53, 101, 0, 255],
    [53, 100, 0, 255],
    [53, 99, 0, 255],
    [52, 97, 0, 255],
    [52, 96, 0, 255],
    [51, 95, 0, 255],
    [51, 94, 0, 255],
    [50, 93, 0, 255],
    [50, 92, 0, 255],
    [49, 91, 0, 255],
    [49, 90, 0, 255],
    [49, 89, 0, 255],
    [48, 88, 0, 255],
    [48, 86, 0, 255],
    [47, 85, 0, 255],
    [47, 84, 0, 255],
    [46, 83, 0, 255],
    [46, 82, 0, 255],
    [46, 81, 0, 255],
    [45, 80, 0, 255],
    [45, 79, 0, 255],
    [44, 78, 0, 255],
    [44, 76, 0, 255],
    [43, 75, 0, 255],
    [43, 74, 0, 255],
    [42, 73, 0, 255],
    [42, 72, 0, 255],
    [42, 71, 0, 255],
    [41, 70, 0, 255],
    [41, 69, 0, 255],
], dtype=np.uint8)


cmaps = {
    'seismic': SEISMIC,
    'coolwarm': COOLWARM,
    'ylgn': YLGN,
    'copper': COPPER,
    'rainbow': RAINBOW,
}


var2cmap = {
    'temperature': 'coolwarm',
    'geopotential': 'coolwarm',
    'z500': 'coolwarm',
    'z1000': 'coolwarm',
    'tau': 'coolwarm',
    't850': 'coolwarm',
    'tcc': 'coolwarm',
    't2m': 'coolwarm',
    'tisr': 'coolwarm',
    'u500': 'coolwarm',
    'u1000': 'coolwarm',
    'v500': 'coolwarm',
    'v1000': 'coolwarm',
}
