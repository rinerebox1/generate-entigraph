# Copyright 2022 Recruit Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import math
from itertools import combinations

from codableopt import Problem, Objective, CategoryVariable, OptSolver, \
    PenaltyAdjustmentMethod


# 時間帯（出発したからのトータル距離の値範囲）によって距離が変化するTSP問題
# ルート/ポイント数を設定
PLACE_NUM = 30
# ルート名を生成
destination_names = [f'destination_{no}' for no in range(PLACE_NUM)]
# ポイント名を生成
place_names = [f'P{no}' for no in range(PLACE_NUM)]


# 距離生成関数
def generate_distances(args_place_names):
    # ポイント間の距離を生成
    tmp_coordinates = {}
    for x in ['start'] + args_place_names:
        tmp_coordinates[x] = (random.randint(1, 1000), random.randint(1, 1000))

    generated_distances = {}
    for point_to_point in combinations(['start'] + args_place_names, 2):
        coordinate_a = tmp_coordinates[point_to_point[0]]
        coordinate_b = tmp_coordinates[point_to_point[1]]
        distance_value = math.sqrt(math.pow(coordinate_a[0] - coordinate_b[0], 2) +
                                   math.pow(coordinate_a[1] - coordinate_b[1], 2))
        generated_distances[point_to_point] = distance_value
        generated_distances[tuple(reversed(point_to_point))] = distance_value
    for x in ['start'] + args_place_names:
        generated_distances[(x, x)] = 0

    return generated_distances


# 朝の時間帯（Startからの出発地点までの合計距離が、0以上300以下の間）におけるポイント間の距離を生成
morning_distances = generate_distances(place_names)
# 昼の時間帯（Startからの出発地点までの合計距離が、301以上700以下の間）におけるポイント間の距離を生成
noon_distances = generate_distances(place_names)
# 夜の時間帯（Startからの出発地点までの合計距離が、701以上）におけるポイント間の距離を生成
night_distances = generate_distances(place_names)

# ルート変数を定義
destinations = [CategoryVariable(name=destination_name, categories=place_names)
                for destination_name in destination_names]

# 問題を設定
problem = Problem(is_max_problem=False)


# 目的関数として、距離を計算する関数を定義
def calc_distance(
        var_destinations,
        para_morning_distances,
        para_noon_distances,
        para_night_distances):
    distance = 0

    for place_from, place_to in zip(
            ['start'] + var_destinations, var_destinations + ['start']):
        # 朝の時間帯（Startからの出発地点までの合計距離が、0以上300以下の間）
        if distance <= 300:
            distance += para_morning_distances[(place_from, place_to)]
        # 昼の時間帯（Startからの出発地点までの合計距離が、301以上700以下の間）
        elif distance <= 700:
            distance += para_noon_distances[(place_from, place_to)]
        # 夜の時間帯（Startからの出発地点までの合計距離が、701以上）
        else:
            distance += para_night_distances[(place_from, place_to)]

    return distance


# 目的関数を定義
problem += Objective(objective=calc_distance,
                     args_map={'var_destinations': destinations,
                               'para_morning_distances': morning_distances,
                               'para_noon_distances': noon_distances,
                               'para_night_distances': night_distances})

# 必ず1度以上、全てのポイントに到達する制約式を追加
for place_name in place_names:
    problem += sum([(destination == place_name)
                   for destination in destinations]) >= 1

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=50000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
root = ["start"] + [answer[x] for x in destination_names] + ["start"]
print(f'root: {" -> ".join(root)}')