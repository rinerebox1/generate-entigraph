Title: Installation — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/

Markdown Content:
[codableopt](https://codable-model-optimizer.readthedocs.io/ja/latest/#)

Use pip[](https://codable-model-optimizer.readthedocs.io/ja/latest/#use-pip "この見出しへのパーマリンク")
---------------------------------------------------------------------------------------------

$ pip install codableopt

Use setup.py[](https://codable-model-optimizer.readthedocs.io/ja/latest/#use-setup-py "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------

# Master branch
$ git clone https://github.com/recruit-tech/codable-model-optimizer
$ python3 setup.py install

Sample:[](https://codable-model-optimizer.readthedocs.io/ja/latest/#sample "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------

import numpy as np
from codableopt import *

# set problem
problem = Problem(is_max_problem=True)

# define variables
x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])

# define objective function
def objective_function(var_x, var_y, var_z, parameters):
    obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        obj_value += 10.0
    elif var_z == 'b':
        obj_value += 8.0
    else:
        # var_z == 'c'
        obj_value -= 3.0

    return obj_value

# set objective function and its arguments
problem += Objective(objective=objective_function,
                     args_map={'var_x': x,
                               'var_y': y,
                               'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

# define constraint
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

print(problem)

solver = OptSolver()

# generate optimization methods to be used within the solver
method = PenaltyAdjustmentMethod(steps=40000)

answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')



Title: Getting Started — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html

Markdown Content:
Installation[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#installation "この見出しへのパーマリンク")
----------------------------------------------------------------------------------------------------------------------------------

### Use pip[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#use-pip "この見出しへのパーマリンク")

$ pip install codableopt

### Use setup.py[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#use-setup-py "この見出しへのパーマリンク")

# Master branch
$ git clone https://github.com/recruit-tech/codable-model-optimizer
$ python3 setup.py install

Basic Usage[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#basic-usage "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------------------------------------------

1.   **問題を設定**

問題オブジェクトを生成する際に、最大化または最小化問題のどちらかを指定をする必要があります。is_max_problemが、Trueの場合は最大化問題、Falseの場合は最小化問題となります。

>>> from codableopt import Problem
>>> problem = Problem(is_max_problem=True)

1.   **変数を定義**

利用する変数を定義します。生成した変数オブジェクトは、制約式や目的関数の引数に利用することができます。

>>> from codableopt import IntVariable, DoubleVariable, CategoryVariable
>>> x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
>>> y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
>>> z = CategoryVariable(name='z', categories=['a', 'b', 'c'])

変数は、内包表記やfor文によってまとめて定義することもできます。

>>> from codableopt import IntVariable
>>> x = [IntVariable(name=f'x_{no}', lower=None, upper=None) for no in range(100)]

1.   **目的関数を設定**

目的関数を問題に設定します。目的関数は、Objectiveオブジェクトを問題オブジェクトに加えることによって、設定できます。Objectiveオブジェクトを生成時には、「目的関数を計算するPython関数」と「引数のマッピング情報」を引数に設定します。 「引数のマッピング情報」は、Dict型で設定し、keyは目的関数の引数名、valueは変数オブジェクトまたは定数やPythonオブジェクトなどを指定します。なお、引数にマッピングした変数オブジェクトは、目的関数を計算するPython関数内では、最適化計算中の変数の値に変換されてから、引数に渡されます。

>>> def objective_function(var_x, var_y, var_z, parameters):
>>>     obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y
>>>     if var_z == 'a':
>>>         obj_value += 10.0
>>>     elif var_z == 'b':
>>>         obj_value += 8.0
>>>     else:
>>>         # var_z == 'c'
>>>         obj_value -= 3.0
>>>
>>>     return obj_value
>>>
>>> problem += Objective(objective=objective_function,
>>>                      args_map={'var_x': x, 'var_y': y, 'var_z': z,
>>>                                'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

「引数のマッピング情報」には、変数リストを渡すこともできます。

>>> from codableopt import IntVariable
>>> x = [IntVariable(name=f'x_{no}', lower=None, upper=None) for no in range(100)]
>>>
>>> problem += Objective(objective=objective_function, args_map={'var_x': x}})

1.   **制約式を定義**

制約式を問題に設定します。制約は、制約式オブジェクトを問題オブジェクトに加えることによって、設定できます。制約式オブジェクトは、変数オブジェクトと不等式を組み合わせることによって生成できます。不等式には、<,<=,>,>=,==が利用できます。また、1次式の制約式しか利用できません。

>>> constant = 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
>>> problem += constant

1.   **最適化計算を実行**

ソルバーオブジェクトと最適化手法オブジェクトを生成し、ソルバーオブジェクトに問題オブジェクトと最適化手法オブジェクトを渡し、最適化計算を行います。ソルバーは、得られた最も良い解と得られた解が制約を全て満たすかの判定フラグを返します。

>>> solver = OptSolver(round_times=2)
>>> method = PenaltyAdjustmentMethod(steps=40000)
>>> answer, is_feasible = solver.solve(problem, method)
>>> print(f'answer:{answer}')
answer:{'x': 0, 'y': 1.5, 'z': 'a'}
>>> print(f'answer_is_feasible:{is_feasible}')
answer_is_feasible:True

Variable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#variable "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------------------------------------

整数・連続値・カテゴリの3種類の変数を提供しています。各変数は、目的関数に渡す引数や制約式に利用します。どの種類の変数も共通で、変数名を設定することができます。変数名は、最適化の解を返す際に利用されます。

### IntVariable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#intvariable "この見出しへのパーマリンク")

IntVariableは、整数型の変数です。lowerには下界値、upperには上界値を設定します。なお、境界値は可能な値として設定されます。また、Noneを設定した場合は、下界値/上界値が設定されます。IntVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

from codableopt import IntVariable
x = IntVariable(name='x', lower=0, upper=None)

### DoubleVariable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#doublevariable "この見出しへのパーマリンク")

DoubleVariableは、連続値型の変数です。lowerには下界値、upperには上界値を設定します。なお、境界値は可能な値として設定されます。また、Noneを設定した場合は、下界値/上界値が設定されます。DoubleVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

from codableopt import DoubleVariable
x = DoubleVariable(name='x', lower=None, upper=2.3)

### CategoryVariable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#categoryvariable "この見出しへのパーマリンク")

CategoryVariableは、カテゴリ型の変数です。categoriesには、取り得るカテゴリ値を設定します。CategoryVariableは、目的関数に渡すことはできるが、制約式に利用することはできません。カテゴリ値を制約式に利用したい場合は、CategoryCaseVariableを利用する必要があります。

from codableopt import CategoryVariable
x = CategoryVariable(name='x', categories=['a', 'b', 'c'])

CategoryCaseVariableは、カテゴリ型の変数と等式を組み合わせることで生成できます。Tupleを利用すれば、比較する値を複数設定でき、いずれかと等しい場合は1、それ以外の場合は0となります。CategoryCaseVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

# xが'a'の時は1、'b'または'c'の時は0となるCategoryCaseVariable
x_a = x == 'a'
# xがb'または'c'の時は1、'a'の時は0となるCategoryCaseVariable
x_bc = x == ('b', 'c')



Title: Advanced Usage — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html

Markdown Content:
Delta Objective Function[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#delta-objective-function "この見出しへのパーマリンク")
---------------------------------------------------------------------------------------------------------------------------------------------------------

目的関数の計算は、関数によっては非常に計算コストが高くなります。しかし、差分計算を用いることで目的関数の計算コストを下げることができます。本ソルバーでも、目的関数の差分計算関数を設定することができます。差分計算は、Objectiveオブジェクト生成時の引数にdelta_objectiveを設定することで利用できます。なお、差分計算関数の引数には、目的関数と同様の引数に加えて、遷移前の変数の値が元の変数名の前にpre_をつけた名前で渡されます。

x = IntVariable(name='x', lower=np.double(0.0), upper=None)
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)

# 目的関数
def obj_fun(var_x, var_y):
    return 3 * var_x + 5 * var_y

# 目的関数の差分計算用の関数
def delta_obj_fun(pre_var_x, pre_var_y, var_x, var_y, parameters):
    delta_value = 0
    if pre_var_x != var_x:
        delta_value += parameters['coef_x'] * (var_x - pre_var_x)
    if pre_var_y != var_y:
        delta_value += parameters['coef_y'] * (var_y - pre_var_y)
    return delta_value

# 目的関数を定義
problem += Objective(objective=obj_fun,
                     delta_objective=delta_obj_fun,
                     args_map={'var_x': x, 'var_y': y,
                               'parameters': {'coef_x': 3.0, 'coef_y': 2.0}})

Custom Optimization Method[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#custom-optimization-method "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------------------------------------------------------------

本ソルバーは、共通アルゴリズム上で最適化手法をカスタマイズして利用することはできます。最適化手法をカスタマイズする場合は、本ソルバーが提供しているOptimizerMethodを継承して実装することで実現することができます。本ソルバーが提供しているペナルティ係数調整手法もその枠組み上で実装されています。

### OptimizerMethod[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#optimizermethod "この見出しへのパーマリンク")

### Sample Code[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#sample-code "この見出しへのパーマリンク")

from typing import List
from random import choice

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
from codableopt.solver.optimizer.optimization_state import OptimizationState

class SampleMethod(OptimizerMethod):

    def  __init__ (self, steps: int):
        super(). __init__ (steps)

    def name(self) -> str:
        return 'sample_method'

    def initialize_of_step(self, state: OptimizationState, step: int):
        # ステップ開始時の処理なし
        pass

    def propose(self, state: OptimizationState, step: int) -> List[ProposalToMove]:
        # 変数から1つランダムに選択する
        solver_variable = choice(state.problem.variables)
        # 選択した変数をランダムに移動する解の遷移を提案する
        return solver_variable.propose_random_move(state)

    def judge(self, state: OptimizationState, step: int) -> bool:
        # 遷移前と遷移後のスコアを比較
        delta_energy = state.current_score.score - state.previous_score.score
        # ソルバー内はエネルギーが低い方が最適性が高いことを表している
        # マイナスの場合に解が改善しているため、提案を受け入れる
        return delta_energy < 0

    def finalize_of_step(self, state: OptimizationState, step: int):
        # ステップ終了時の処理なし
        pass

[deprecation] User Define Constraint[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#deprecation-user-define-constraint "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

非推奨ではありますが、本ソルバーでは、制約式を関数として渡すこともできます。関数の返り値には制約違反量を設定します。引数は、目的関数同様にargs_mapを設定することで指定できます。ただし、デフォルトで提供しているmethodでは、User Define Constraintを利用している最適化問題は実用に耐えうる最適化精度を実現できません。そのため現状では利用することは推奨していません。

### Sample Code[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#id1 "この見出しへのパーマリンク")

# 制約式を定義
def udf_constraint_function(var_x, var_y, var_z):
    violation_amount = 2 * var_x + 4 * var_y - 8
    if var_z == 'a':
        violation_amount += 2
    else:
        violation_amount += 3

    if violation_amount <= 0:
        return 0
    else:
        return violation_amount

constant = UserDefineConstraint(udf_constraint_function,
                                args_map={'var_x': x, 'var_y': y, 'var_z': z},
                                constraint_name='user_define_constraint')
problem += constant



Title: Algorithm — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html

Markdown Content:
本ソルバーの最適化アルゴリズムは、「全手法で共通のアルゴリズム部分」と「各手法で異なるアルゴリズム部分」に分かれています。「全手法で共通のアルゴリズム部分」はカスタマイズできませんが、「各手法で異なるアルゴリズム部分」はカスタムマイズすることができます。また、本ソルバーでは、ペナルティ係数調整手法という名称で最適化手法を提供しています。

Common Algorithm[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#common-algorithm "この見出しへのパーマリンク")
------------------------------------------------------------------------------------------------------------------------------------

### Algorithm of OptSolver[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#algorithm-of-optsolver "この見出しへのパーマリンク")

共通アルゴリズムは、下記のようなステップのアルゴリズムです。

1.   **初期解の生成**

num_to_select_init_answerで指定した数、ランダムな解を生成します。ランダムな解は、変数毎に上界/下界またはカテゴリ値からランダムに選択した値です。生成したランダムな解を各変数のスケールを正規化し、簡易的なアルゴリズムによって選択した解群間のユーグリッド距離の合計が最大となるようなround_times個の解群を初期解群として採用します。ただし、引数のinit_answersに初期解が設定されている場合は、ランダムな解を生成せずに、指定された初期解を利用します。

1.   **最適化の実行**

採用した初期解毎に最適化の実行します。また、このときn_jobs引数によって並列処理の実行を設定している場合は、初期解毎に並列処理が実行されます。全ての初期解に対する最適化が完了したら、返された最適化処理の結果から実行可能解がある場合はその中から最も目的関数の値が最も良い解を、実行可能解がない場合はその中から目的関数の値に制約違反ペナルティを加えた値が最も良い解を選択して、最適化の実行結果として返します。（制約違反ペナルティのペナルティ係数は、共通ではなく、各最適化処理内で決定している値を利用する。結果、フェアな比較ではないが、実行可能解がない場合の解は参考程度の解なので、現状このような仕様となっている。また今後、細かな結果など含めて最適化結果を返せるようなインタフェースを検討する予定です。。）

### Algorithm of Optimizer[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#algorithm-of-optimizer "この見出しへのパーマリンク")

1.   **ペナルティ係数の調整**

初期解（最適化試行）毎に、answer_num_to_tune_penaltyで指定した数、ランダムな解を生成します。ランダムな解は、変数毎に上界/下界またはカテゴリ値からランダムに選択した値とする。生成したランダムな解群から各制約式の違反量の違反量を計算し、「生成したランダムな解群における目的関数の最大値と最小値の差分のpenalty_strength倍」と「各制約式のペナルティスコア」が等しくなるような各制約式のペナルティ係数に調整します。

1.   **初期解のスコア計算**

初期解から目的関数の値と制約違反のペナルティの値を計算し、合算して初期解のスコアとして採用します。また、初期解を現状の最適解として採択します。

1.   **method.initialize_of_step実施**

設定したmethodのinitialize_of_step関数を呼び出します。

1.   **method.propose実施**

設定したmethodのpropose関数を呼び出し、解の遷移案を取得します。

1.   **提案された解のスコア計算**

解の遷移案を実行した場合のスコア（目的関数の値と制約違反のペナルティの値の合算）を計算します。

1.   **method.judge実施**

設定したmethodのjudge関数を呼び出し、解の遷移を実施有無を決定します。解の遷移が決定した場合は、現状の解を遷移させる。

1.   **最適性確認**

6で解が遷移した場合は、新たな解の最適性の確認を行います。最適解は、実行可能解を優先し、その上でスコアが良い方を優先するように選択します。現状の最適解より良い場合は、最適解を変更します。

1.   **method.finalize_of_step実施**

設定したmethodのfinalize_of_step実施関数を呼び出します。

1.   **終了判定**

3-8を繰り返した回数を計算する。methodで設定したsteps数に達した場合は、終了とし、現状の最適解を返す。達していない場合は、3に戻り、同様に処理を繰り返していきます。

Method[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#method "この見出しへのパーマリンク")
----------------------------------------------------------------------------------------------------------------

### Penalty Adjustment Method[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#penalty-adjustment-method "この見出しへのパーマリンク")

1.   **method.initialize_of_step**

ステップ開始時の処理はありません。

1.   **method.propose**

random_movement_rateの確率で、ランダム遷移を提案する。また、(1-random_movement_rate)の確率で、ペナルティが小さくなる遷移を提案します。 ランダム遷移を提案する時は、最適化問題内の1つの変数をランダムに選択し、上界から下界値またはカテゴリ値から1つの値を選択し、提案します。ただし、変数が数値かつデータ遷移履歴がhistory_value_size以上のデータ件数の場合は、対象変数のデータ遷移履歴値の平均値と標準偏差値を計算し、「平均値 - 標準偏差値 * range_std_rate」から「平均値 + 標準偏差値 * range_std_rate」までの値の範囲からランダムで値を選択して、提案します。またペナルティが小さくなる遷移を提案する時は、最適化問題内の1つの変数をランダムに選択し、その変数を動かすことでペナルティが減るような値を計算によって求め、提案します。また、ペナルティが最小になる複数の値が存在した場合は、その中からランダムに選択し、提案します。（＊この計算において、UserDefineConstraintは対象になりません。）

1.   **method.judge**

スコアを比較し、現状の解より良い場合は遷移するという判定結果を、それ以外の場合は遷移しないという判定結果を返します。

1.   **method.finalize_of_step**

「最後に解が遷移してから経過したStep数」と「最後にペナルティ係数を更新してから経過したStep数」を計算し、小さな方の値を採用し、その値がsteps_while_not_improve_score以上に達していたらペナルティ係数を調整します。 現状の解が実行可能解である場合は、ペナルティ係数に「現状の解のペナルティ違反量を正規化した値」と(1 + self._delta_penalty_rate)を積算し、ペナルティ係数をあげます。また、現状の解が実行可能解ではない場合は、一律にペナルティ係数に(1 - delta_penalty_rate)を積算し、ペナルティ係数を下げます。



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

import numpy as np

from codableopt import Problem, Objective, IntVariable, DoubleVariable, \
    CategoryVariable, OptSolver, PenaltyAdjustmentMethod

# 変数を定義
x = IntVariable(name='x', lower=np.double(0.0), upper=np.double(2))
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


# 目的関数に指定する関数
def objective_function(var_x, var_y, var_z, parameters):
    obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        obj_value += 10.0
    elif var_z == 'b':
        obj_value += 8.0
    else:
        # var_z == 'c'
        obj_value -= 3.0

    return obj_value


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x, 'var_y': y, 'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

# 制約式を定義
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

# 問題を確認
print(problem)

# ソルバーを生成
solver = OptSolver()

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=40000)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')

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

import numpy as np

from codableopt import Problem, Objective, IntVariable, DoubleVariable, \
    CategoryVariable, OptSolver, PenaltyAdjustmentMethod

# 変数を定義
x = IntVariable(name='x', lower=0, upper=None)
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


# 目的関数に指定する関数
def objective_function(var_x, var_y, var_z, parameters):
    value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        value += 10.0
    elif var_z == 'b':
        value += 8.0
    else:
        # var_z == 'c'
        value -= 3.0

    return value


# 差分計算による目的関数に指定する関数
def delta_objective_function(
        pre_var_x,
        pre_var_y,
        pre_var_z,
        var_x,
        var_y,
        var_z,
        parameters):
    delta_value = 0
    if pre_var_x != var_x:
        delta_value += parameters['coef_x'] * (var_x - pre_var_x)

    if pre_var_y != var_y:
        delta_value += parameters['coef_y'] * (var_y - pre_var_y)

    if pre_var_z != var_z:
        if pre_var_z == 'a':
            delta_value -= 10.0
        elif pre_var_z == 'b':
            delta_value -= 8.0
        else:
            # pre_z == 'c'
            delta_value += 3.0

        if var_z == 'a':
            delta_value += 10.0
        elif var_z == 'b':
            delta_value += 8.0
        else:
            # z == 'c'
            delta_value -= 3.0

    return delta_value


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x, 'var_y': y, 'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}},
                     delta_objective=delta_objective_function)

# 制約式を定義
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

# ソルバーを生成
solver = OptSolver(round_times=2, debug=True, debug_unit_step=1000)

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=40000)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')

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

from codableopt import Problem, Objective, IntVariable, OptSolver, \
    PenaltyAdjustmentMethod


# アイテム数、最大重量を設定
item_num = 40
max_weight = 1000
# アイテム名を生成
item_names = [f'item_{no}' for no in range(item_num)]
# アイテムのバリューと重量を設定
parameter_item_values = [random.randint(10, 50) for _ in item_names]
parameter_item_weights = [random.randint(20, 40) for _ in item_names]

# アイテムのBool変数を定義
var_item_flags = [IntVariable(name=item_name, lower=0, upper=1) for item_name in item_names]


# 目的関数として、距離を計算する関数を定義
def calculate_total_values(item_flags, item_values):
    return sum([flag * value for flag, value in zip(item_flags, item_values)])


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=calculate_total_values,
                     args_map={'item_flags': var_item_flags, 'item_values': parameter_item_values})

# 重量制限の制約式を追加
problem += sum([item_flag * weight for item_flag,
                weight in zip(var_item_flags,
                              parameter_item_weights)]) <= max_weight

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=10000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'select_items: 'f'{", ".join([x for x in answer.keys() if answer[x] == 1])}')

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

from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


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


# 単純なTSP問題
# ルート/ポイント数を設定
PLACE_NUM = 30
# 行き先名を生成
destination_names = [f'destination_{no}' for no in range(PLACE_NUM)]
# ポイント名を生成
place_names = [f'P{no}' for no in range(PLACE_NUM)]
# 距離を生成
distances = generate_distances(place_names)

# ルート変数を定義
destinations = [CategoryVariable(name=destination_name, categories=place_names)
                for destination_name in destination_names]

# 問題を設定
problem = Problem(is_max_problem=False)


# 目的関数として、距離を計算する関数を定義
def calc_distance(var_destinations, para_distances):
    return sum([para_distances[(x, y)] for x, y
                in zip(['start'] + var_destinations, var_destinations + ['start'])])


# 目的関数を定義
problem += Objective(objective=calc_distance,
                     args_map={'var_destinations': destinations, 'para_distances': distances})

# 必ず1度以上、全てのポイントに到達する制約式を追加
for place_name in place_names:
    problem += sum([(destination == place_name)
                   for destination in destinations]) >= 1

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=50000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
root = ['start'] + [answer[root] for root in destination_names] + ['start']
print(f'root: {" -> ".join(root)}')

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

import numpy as np

from codableopt import Problem, Objective, IntVariable, OptSolver, \
    PenaltyAdjustmentMethod


# 顧客数、CM数を設定
CUSTOMER_NUM = 1000
CM_NUM = 100
SELECTED_CM_LIMIT = 10
# 顧客がCMを見る確率を生成
view_rates = np.random.rand(CUSTOMER_NUM, CM_NUM) / 10 / SELECTED_CM_LIMIT

# CMの放送有無の変数を定義
cm_times = [IntVariable(name=f'cm_{no}', lower=0, upper=1) for no in range(CM_NUM)]

# 問題を設定
problem = Problem(is_max_problem=True)


# 目的関数として、CMを1度でも見る確率を計算
def calculate_view_rate_sum(var_cm_times, para_non_view_rates):
    selected_cm_noes = \
        [cm_no for cm_no, var_cm_time in enumerate(var_cm_times) if var_cm_time == 1]
    view_rate_per_customers = np.ones(para_non_view_rates.shape[0]) \
        - np.prod(para_non_view_rates[:, selected_cm_noes], axis=1)
    return np.sum(view_rate_per_customers)


# 目的関数を定義
problem += Objective(objective=calculate_view_rate_sum,
                     args_map={'var_cm_times': cm_times,
                               'para_non_view_rates': np.ones(view_rates.shape) - view_rates})

# CMの選択数の制約式を追加
problem += sum(cm_times) <= SELECTED_CM_LIMIT

print(problem)

# 最適化実施
solver = OptSolver(round_times=2, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=100000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'selected cm: {[cm_name for cm_name in answer.keys() if answer[cm_name] == 1]}')

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

from typing import List

import pandas as pd

from sample.usage.problem.matching_problem_generator import MatchingProblemGenerator
from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


# 最適化問題定義
print('generate Problem')
matching = MatchingProblemGenerator.generate(customer_num=1000, item_num=20)

print('start Optimization')
# 変数を定義
selected_items = \
    [CategoryVariable(name=f'item_for_{customer.name}', categories=matching.item_names)
     for customer in matching.customers]
selected_coupons = \
    [CategoryVariable(name=f'coupon_for_{customer.name}', categories=matching.coupon_names)
     for customer in matching.customers]


# 利益の期待値を計算する関数、目的関数に利用
def calculate_benefit(
        var_selected_items: List[str],
        var_selected_coupons: List[str],
        para_customer_features_df: pd.DataFrame,
        para_item_features_df: pd.DataFrame,
        para_coupon_features_df: pd.DataFrame,
        para_buy_rate_model):

    features_df = pd.concat([
        para_customer_features_df.reset_index(drop=True),
        para_item_features_df.loc[var_selected_items, :].reset_index(drop=True),
        para_coupon_features_df.loc[var_selected_coupons, :].reset_index(drop=True)
    ], axis=1)

    # 目的関数内で機械学習モデルを利用
    buy_rate = \
        [x[1] for x in para_buy_rate_model.predict_proba(features_df.drop(columns='item_cost'))]

    return sum(
        buy_rate *
        (features_df['item_price'] - features_df['item_cost'] - features_df['coupon_down_price']))


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=calculate_benefit,
                     args_map={
                         'var_selected_items': selected_items,
                         'var_selected_coupons': selected_coupons,
                         'para_customer_features_df': matching.customer_features_df,
                         'para_item_features_df': matching.item_features_df,
                         'para_coupon_features_df': matching.coupon_features_df,
                         'para_buy_rate_model': matching.buy_rate_model
                     })

# 制約式を定義
for item in matching.items:
    # 必ず1人以上のカスタマーにアイテムを表示する制約式を設定
    problem += sum([(x == item.name) for x in selected_items]) >= 1
    # 同じアイテムの最大表示人数を制限する制約式を設定
    problem += sum([(x == item.name) for x in selected_items]) <= matching.max_display_num_per_item

for coupon in matching.coupons:
    # クーポンの最大発行数の制約式を設定
    problem += \
        sum([(x == coupon.name) for x in selected_coupons]) <= matching.max_display_num_per_coupon


# 最適化実施
print('start solve')
solver = OptSolver(round_times=1, debug=True, debug_unit_step=1000,
                   num_to_tune_penalty=100, num_to_select_init_answer=1)
method = PenaltyAdjustmentMethod(steps=40000)
answer, is_feasible = solver.solve(problem, method)

print(f'answer_is_feasible:{is_feasible}')
print(answer)

Title: Installation — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/

Markdown Content:
[codableopt](https://codable-model-optimizer.readthedocs.io/ja/latest/#)

Use pip[](https://codable-model-optimizer.readthedocs.io/ja/latest/#use-pip "この見出しへのパーマリンク")
---------------------------------------------------------------------------------------------

$ pip install codableopt

Use setup.py[](https://codable-model-optimizer.readthedocs.io/ja/latest/#use-setup-py "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------

# Master branch
$ git clone https://github.com/recruit-tech/codable-model-optimizer
$ python3 setup.py install

Sample:[](https://codable-model-optimizer.readthedocs.io/ja/latest/#sample "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------

import numpy as np
from codableopt import *

# set problem
problem = Problem(is_max_problem=True)

# define variables
x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])

# define objective function
def objective_function(var_x, var_y, var_z, parameters):
    obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        obj_value += 10.0
    elif var_z == 'b':
        obj_value += 8.0
    else:
        # var_z == 'c'
        obj_value -= 3.0

    return obj_value

# set objective function and its arguments
problem += Objective(objective=objective_function,
                     args_map={'var_x': x,
                               'var_y': y,
                               'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

# define constraint
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

print(problem)

solver = OptSolver()

# generate optimization methods to be used within the solver
method = PenaltyAdjustmentMethod(steps=40000)

answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')



Title: Getting Started — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html

Markdown Content:
Installation[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#installation "この見出しへのパーマリンク")
----------------------------------------------------------------------------------------------------------------------------------

### Use pip[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#use-pip "この見出しへのパーマリンク")

$ pip install codableopt

### Use setup.py[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#use-setup-py "この見出しへのパーマリンク")

# Master branch
$ git clone https://github.com/recruit-tech/codable-model-optimizer
$ python3 setup.py install

Basic Usage[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#basic-usage "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------------------------------------------

1.   **問題を設定**

問題オブジェクトを生成する際に、最大化または最小化問題のどちらかを指定をする必要があります。is_max_problemが、Trueの場合は最大化問題、Falseの場合は最小化問題となります。

>>> from codableopt import Problem
>>> problem = Problem(is_max_problem=True)

1.   **変数を定義**

利用する変数を定義します。生成した変数オブジェクトは、制約式や目的関数の引数に利用することができます。

>>> from codableopt import IntVariable, DoubleVariable, CategoryVariable
>>> x = IntVariable(name='x', lower=np.double(0), upper=np.double(5))
>>> y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
>>> z = CategoryVariable(name='z', categories=['a', 'b', 'c'])

変数は、内包表記やfor文によってまとめて定義することもできます。

>>> from codableopt import IntVariable
>>> x = [IntVariable(name=f'x_{no}', lower=None, upper=None) for no in range(100)]

1.   **目的関数を設定**

目的関数を問題に設定します。目的関数は、Objectiveオブジェクトを問題オブジェクトに加えることによって、設定できます。Objectiveオブジェクトを生成時には、「目的関数を計算するPython関数」と「引数のマッピング情報」を引数に設定します。 「引数のマッピング情報」は、Dict型で設定し、keyは目的関数の引数名、valueは変数オブジェクトまたは定数やPythonオブジェクトなどを指定します。なお、引数にマッピングした変数オブジェクトは、目的関数を計算するPython関数内では、最適化計算中の変数の値に変換されてから、引数に渡されます。

>>> def objective_function(var_x, var_y, var_z, parameters):
>>>     obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y
>>>     if var_z == 'a':
>>>         obj_value += 10.0
>>>     elif var_z == 'b':
>>>         obj_value += 8.0
>>>     else:
>>>         # var_z == 'c'
>>>         obj_value -= 3.0
>>>
>>>     return obj_value
>>>
>>> problem += Objective(objective=objective_function,
>>>                      args_map={'var_x': x, 'var_y': y, 'var_z': z,
>>>                                'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

「引数のマッピング情報」には、変数リストを渡すこともできます。

>>> from codableopt import IntVariable
>>> x = [IntVariable(name=f'x_{no}', lower=None, upper=None) for no in range(100)]
>>>
>>> problem += Objective(objective=objective_function, args_map={'var_x': x}})

1.   **制約式を定義**

制約式を問題に設定します。制約は、制約式オブジェクトを問題オブジェクトに加えることによって、設定できます。制約式オブジェクトは、変数オブジェクトと不等式を組み合わせることによって生成できます。不等式には、<,<=,>,>=,==が利用できます。また、1次式の制約式しか利用できません。

>>> constant = 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
>>> problem += constant

1.   **最適化計算を実行**

ソルバーオブジェクトと最適化手法オブジェクトを生成し、ソルバーオブジェクトに問題オブジェクトと最適化手法オブジェクトを渡し、最適化計算を行います。ソルバーは、得られた最も良い解と得られた解が制約を全て満たすかの判定フラグを返します。

>>> solver = OptSolver(round_times=2)
>>> method = PenaltyAdjustmentMethod(steps=40000)
>>> answer, is_feasible = solver.solve(problem, method)
>>> print(f'answer:{answer}')
answer:{'x': 0, 'y': 1.5, 'z': 'a'}
>>> print(f'answer_is_feasible:{is_feasible}')
answer_is_feasible:True

Variable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#variable "この見出しへのパーマリンク")
--------------------------------------------------------------------------------------------------------------------------

整数・連続値・カテゴリの3種類の変数を提供しています。各変数は、目的関数に渡す引数や制約式に利用します。どの種類の変数も共通で、変数名を設定することができます。変数名は、最適化の解を返す際に利用されます。

### IntVariable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#intvariable "この見出しへのパーマリンク")

IntVariableは、整数型の変数です。lowerには下界値、upperには上界値を設定します。なお、境界値は可能な値として設定されます。また、Noneを設定した場合は、下界値/上界値が設定されます。IntVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

from codableopt import IntVariable
x = IntVariable(name='x', lower=0, upper=None)

### DoubleVariable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#doublevariable "この見出しへのパーマリンク")

DoubleVariableは、連続値型の変数です。lowerには下界値、upperには上界値を設定します。なお、境界値は可能な値として設定されます。また、Noneを設定した場合は、下界値/上界値が設定されます。DoubleVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

from codableopt import DoubleVariable
x = DoubleVariable(name='x', lower=None, upper=2.3)

### CategoryVariable[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/getting_started.html#categoryvariable "この見出しへのパーマリンク")

CategoryVariableは、カテゴリ型の変数です。categoriesには、取り得るカテゴリ値を設定します。CategoryVariableは、目的関数に渡すことはできるが、制約式に利用することはできません。カテゴリ値を制約式に利用したい場合は、CategoryCaseVariableを利用する必要があります。

from codableopt import CategoryVariable
x = CategoryVariable(name='x', categories=['a', 'b', 'c'])

CategoryCaseVariableは、カテゴリ型の変数と等式を組み合わせることで生成できます。Tupleを利用すれば、比較する値を複数設定でき、いずれかと等しい場合は1、それ以外の場合は0となります。CategoryCaseVariableは、目的関数に渡す引数と制約式のどちらにも利用できます。

# xが'a'の時は1、'b'または'c'の時は0となるCategoryCaseVariable
x_a = x == 'a'
# xがb'または'c'の時は1、'a'の時は0となるCategoryCaseVariable
x_bc = x == ('b', 'c')



Title: Advanced Usage — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html

Markdown Content:
Delta Objective Function[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#delta-objective-function "この見出しへのパーマリンク")
---------------------------------------------------------------------------------------------------------------------------------------------------------

目的関数の計算は、関数によっては非常に計算コストが高くなります。しかし、差分計算を用いることで目的関数の計算コストを下げることができます。本ソルバーでも、目的関数の差分計算関数を設定することができます。差分計算は、Objectiveオブジェクト生成時の引数にdelta_objectiveを設定することで利用できます。なお、差分計算関数の引数には、目的関数と同様の引数に加えて、遷移前の変数の値が元の変数名の前にpre_をつけた名前で渡されます。

x = IntVariable(name='x', lower=np.double(0.0), upper=None)
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)

# 目的関数
def obj_fun(var_x, var_y):
    return 3 * var_x + 5 * var_y

# 目的関数の差分計算用の関数
def delta_obj_fun(pre_var_x, pre_var_y, var_x, var_y, parameters):
    delta_value = 0
    if pre_var_x != var_x:
        delta_value += parameters['coef_x'] * (var_x - pre_var_x)
    if pre_var_y != var_y:
        delta_value += parameters['coef_y'] * (var_y - pre_var_y)
    return delta_value

# 目的関数を定義
problem += Objective(objective=obj_fun,
                     delta_objective=delta_obj_fun,
                     args_map={'var_x': x, 'var_y': y,
                               'parameters': {'coef_x': 3.0, 'coef_y': 2.0}})

Custom Optimization Method[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#custom-optimization-method "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------------------------------------------------------------

本ソルバーは、共通アルゴリズム上で最適化手法をカスタマイズして利用することはできます。最適化手法をカスタマイズする場合は、本ソルバーが提供しているOptimizerMethodを継承して実装することで実現することができます。本ソルバーが提供しているペナルティ係数調整手法もその枠組み上で実装されています。

### OptimizerMethod[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#optimizermethod "この見出しへのパーマリンク")

### Sample Code[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#sample-code "この見出しへのパーマリンク")

from typing import List
from random import choice

from codableopt.solver.optimizer.entity.proposal_to_move import ProposalToMove
from codableopt.solver.optimizer.method.optimizer_method import OptimizerMethod
from codableopt.solver.optimizer.optimization_state import OptimizationState

class SampleMethod(OptimizerMethod):

    def  __init__ (self, steps: int):
        super(). __init__ (steps)

    def name(self) -> str:
        return 'sample_method'

    def initialize_of_step(self, state: OptimizationState, step: int):
        # ステップ開始時の処理なし
        pass

    def propose(self, state: OptimizationState, step: int) -> List[ProposalToMove]:
        # 変数から1つランダムに選択する
        solver_variable = choice(state.problem.variables)
        # 選択した変数をランダムに移動する解の遷移を提案する
        return solver_variable.propose_random_move(state)

    def judge(self, state: OptimizationState, step: int) -> bool:
        # 遷移前と遷移後のスコアを比較
        delta_energy = state.current_score.score - state.previous_score.score
        # ソルバー内はエネルギーが低い方が最適性が高いことを表している
        # マイナスの場合に解が改善しているため、提案を受け入れる
        return delta_energy < 0

    def finalize_of_step(self, state: OptimizationState, step: int):
        # ステップ終了時の処理なし
        pass

[deprecation] User Define Constraint[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#deprecation-user-define-constraint "この見出しへのパーマリンク")
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

非推奨ではありますが、本ソルバーでは、制約式を関数として渡すこともできます。関数の返り値には制約違反量を設定します。引数は、目的関数同様にargs_mapを設定することで指定できます。ただし、デフォルトで提供しているmethodでは、User Define Constraintを利用している最適化問題は実用に耐えうる最適化精度を実現できません。そのため現状では利用することは推奨していません。

### Sample Code[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/advanced_usage.html#id1 "この見出しへのパーマリンク")

# 制約式を定義
def udf_constraint_function(var_x, var_y, var_z):
    violation_amount = 2 * var_x + 4 * var_y - 8
    if var_z == 'a':
        violation_amount += 2
    else:
        violation_amount += 3

    if violation_amount <= 0:
        return 0
    else:
        return violation_amount

constant = UserDefineConstraint(udf_constraint_function,
                                args_map={'var_x': x, 'var_y': y, 'var_z': z},
                                constraint_name='user_define_constraint')
problem += constant



Title: Algorithm — codableopt v0.1 ドキュメント

URL Source: https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html

Markdown Content:
本ソルバーの最適化アルゴリズムは、「全手法で共通のアルゴリズム部分」と「各手法で異なるアルゴリズム部分」に分かれています。「全手法で共通のアルゴリズム部分」はカスタマイズできませんが、「各手法で異なるアルゴリズム部分」はカスタムマイズすることができます。また、本ソルバーでは、ペナルティ係数調整手法という名称で最適化手法を提供しています。

Common Algorithm[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#common-algorithm "この見出しへのパーマリンク")
------------------------------------------------------------------------------------------------------------------------------------

### Algorithm of OptSolver[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#algorithm-of-optsolver "この見出しへのパーマリンク")

共通アルゴリズムは、下記のようなステップのアルゴリズムです。

1.   **初期解の生成**

num_to_select_init_answerで指定した数、ランダムな解を生成します。ランダムな解は、変数毎に上界/下界またはカテゴリ値からランダムに選択した値です。生成したランダムな解を各変数のスケールを正規化し、簡易的なアルゴリズムによって選択した解群間のユーグリッド距離の合計が最大となるようなround_times個の解群を初期解群として採用します。ただし、引数のinit_answersに初期解が設定されている場合は、ランダムな解を生成せずに、指定された初期解を利用します。

1.   **最適化の実行**

採用した初期解毎に最適化の実行します。また、このときn_jobs引数によって並列処理の実行を設定している場合は、初期解毎に並列処理が実行されます。全ての初期解に対する最適化が完了したら、返された最適化処理の結果から実行可能解がある場合はその中から最も目的関数の値が最も良い解を、実行可能解がない場合はその中から目的関数の値に制約違反ペナルティを加えた値が最も良い解を選択して、最適化の実行結果として返します。（制約違反ペナルティのペナルティ係数は、共通ではなく、各最適化処理内で決定している値を利用する。結果、フェアな比較ではないが、実行可能解がない場合の解は参考程度の解なので、現状このような仕様となっている。また今後、細かな結果など含めて最適化結果を返せるようなインタフェースを検討する予定です。。）

### Algorithm of Optimizer[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#algorithm-of-optimizer "この見出しへのパーマリンク")

1.   **ペナルティ係数の調整**

初期解（最適化試行）毎に、answer_num_to_tune_penaltyで指定した数、ランダムな解を生成します。ランダムな解は、変数毎に上界/下界またはカテゴリ値からランダムに選択した値とする。生成したランダムな解群から各制約式の違反量の違反量を計算し、「生成したランダムな解群における目的関数の最大値と最小値の差分のpenalty_strength倍」と「各制約式のペナルティスコア」が等しくなるような各制約式のペナルティ係数に調整します。

1.   **初期解のスコア計算**

初期解から目的関数の値と制約違反のペナルティの値を計算し、合算して初期解のスコアとして採用します。また、初期解を現状の最適解として採択します。

1.   **method.initialize_of_step実施**

設定したmethodのinitialize_of_step関数を呼び出します。

1.   **method.propose実施**

設定したmethodのpropose関数を呼び出し、解の遷移案を取得します。

1.   **提案された解のスコア計算**

解の遷移案を実行した場合のスコア（目的関数の値と制約違反のペナルティの値の合算）を計算します。

1.   **method.judge実施**

設定したmethodのjudge関数を呼び出し、解の遷移を実施有無を決定します。解の遷移が決定した場合は、現状の解を遷移させる。

1.   **最適性確認**

6で解が遷移した場合は、新たな解の最適性の確認を行います。最適解は、実行可能解を優先し、その上でスコアが良い方を優先するように選択します。現状の最適解より良い場合は、最適解を変更します。

1.   **method.finalize_of_step実施**

設定したmethodのfinalize_of_step実施関数を呼び出します。

1.   **終了判定**

3-8を繰り返した回数を計算する。methodで設定したsteps数に達した場合は、終了とし、現状の最適解を返す。達していない場合は、3に戻り、同様に処理を繰り返していきます。

Method[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#method "この見出しへのパーマリンク")
----------------------------------------------------------------------------------------------------------------

### Penalty Adjustment Method[](https://codable-model-optimizer.readthedocs.io/ja/latest/manual/algorithm.html#penalty-adjustment-method "この見出しへのパーマリンク")

1.   **method.initialize_of_step**

ステップ開始時の処理はありません。

1.   **method.propose**

random_movement_rateの確率で、ランダム遷移を提案する。また、(1-random_movement_rate)の確率で、ペナルティが小さくなる遷移を提案します。 ランダム遷移を提案する時は、最適化問題内の1つの変数をランダムに選択し、上界から下界値またはカテゴリ値から1つの値を選択し、提案します。ただし、変数が数値かつデータ遷移履歴がhistory_value_size以上のデータ件数の場合は、対象変数のデータ遷移履歴値の平均値と標準偏差値を計算し、「平均値 - 標準偏差値 * range_std_rate」から「平均値 + 標準偏差値 * range_std_rate」までの値の範囲からランダムで値を選択して、提案します。またペナルティが小さくなる遷移を提案する時は、最適化問題内の1つの変数をランダムに選択し、その変数を動かすことでペナルティが減るような値を計算によって求め、提案します。また、ペナルティが最小になる複数の値が存在した場合は、その中からランダムに選択し、提案します。（＊この計算において、UserDefineConstraintは対象になりません。）

1.   **method.judge**

スコアを比較し、現状の解より良い場合は遷移するという判定結果を、それ以外の場合は遷移しないという判定結果を返します。

1.   **method.finalize_of_step**

「最後に解が遷移してから経過したStep数」と「最後にペナルティ係数を更新してから経過したStep数」を計算し、小さな方の値を採用し、その値がsteps_while_not_improve_score以上に達していたらペナルティ係数を調整します。 現状の解が実行可能解である場合は、ペナルティ係数に「現状の解のペナルティ違反量を正規化した値」と(1 + self._delta_penalty_rate)を積算し、ペナルティ係数をあげます。また、現状の解が実行可能解ではない場合は、一律にペナルティ係数に(1 - delta_penalty_rate)を積算し、ペナルティ係数を下げます。



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

import numpy as np

from codableopt import Problem, Objective, IntVariable, DoubleVariable, \
    CategoryVariable, OptSolver, PenaltyAdjustmentMethod

# 変数を定義
x = IntVariable(name='x', lower=np.double(0.0), upper=np.double(2))
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


# 目的関数に指定する関数
def objective_function(var_x, var_y, var_z, parameters):
    obj_value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        obj_value += 10.0
    elif var_z == 'b':
        obj_value += 8.0
    else:
        # var_z == 'c'
        obj_value -= 3.0

    return obj_value


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x, 'var_y': y, 'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}})

# 制約式を定義
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

# 問題を確認
print(problem)

# ソルバーを生成
solver = OptSolver()

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=40000)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')

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

import numpy as np

from codableopt import Problem, Objective, IntVariable, DoubleVariable, \
    CategoryVariable, OptSolver, PenaltyAdjustmentMethod

# 変数を定義
x = IntVariable(name='x', lower=0, upper=None)
y = DoubleVariable(name='y', lower=np.double(0.0), upper=None)
z = CategoryVariable(name='z', categories=['a', 'b', 'c'])


# 目的関数に指定する関数
def objective_function(var_x, var_y, var_z, parameters):
    value = parameters['coef_x'] * var_x + parameters['coef_y'] * var_y

    if var_z == 'a':
        value += 10.0
    elif var_z == 'b':
        value += 8.0
    else:
        # var_z == 'c'
        value -= 3.0

    return value


# 差分計算による目的関数に指定する関数
def delta_objective_function(
        pre_var_x,
        pre_var_y,
        pre_var_z,
        var_x,
        var_y,
        var_z,
        parameters):
    delta_value = 0
    if pre_var_x != var_x:
        delta_value += parameters['coef_x'] * (var_x - pre_var_x)

    if pre_var_y != var_y:
        delta_value += parameters['coef_y'] * (var_y - pre_var_y)

    if pre_var_z != var_z:
        if pre_var_z == 'a':
            delta_value -= 10.0
        elif pre_var_z == 'b':
            delta_value -= 8.0
        else:
            # pre_z == 'c'
            delta_value += 3.0

        if var_z == 'a':
            delta_value += 10.0
        elif var_z == 'b':
            delta_value += 8.0
        else:
            # z == 'c'
            delta_value -= 3.0

    return delta_value


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=objective_function,
                     args_map={'var_x': x, 'var_y': y, 'var_z': z,
                               'parameters': {'coef_x': -3.0, 'coef_y': 4.0}},
                     delta_objective=delta_objective_function)

# 制約式を定義
problem += 2 * x + 4 * y + 2 * (z == 'a') + 3 * (z == ('b', 'c')) <= 8
problem += 2 * x - y + 2 * (z == 'b') > 3

# ソルバーを生成
solver = OptSolver(round_times=2, debug=True, debug_unit_step=1000)

# ソルバー内で使う最適化手法を生成
method = PenaltyAdjustmentMethod(steps=40000)

# 最適化実施
answer, is_feasible = solver.solve(problem, method)
print(f'answer:{answer}, answer_is_feasible:{is_feasible}')

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

from codableopt import Problem, Objective, IntVariable, OptSolver, \
    PenaltyAdjustmentMethod


# アイテム数、最大重量を設定
item_num = 40
max_weight = 1000
# アイテム名を生成
item_names = [f'item_{no}' for no in range(item_num)]
# アイテムのバリューと重量を設定
parameter_item_values = [random.randint(10, 50) for _ in item_names]
parameter_item_weights = [random.randint(20, 40) for _ in item_names]

# アイテムのBool変数を定義
var_item_flags = [IntVariable(name=item_name, lower=0, upper=1) for item_name in item_names]


# 目的関数として、距離を計算する関数を定義
def calculate_total_values(item_flags, item_values):
    return sum([flag * value for flag, value in zip(item_flags, item_values)])


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=calculate_total_values,
                     args_map={'item_flags': var_item_flags, 'item_values': parameter_item_values})

# 重量制限の制約式を追加
problem += sum([item_flag * weight for item_flag,
                weight in zip(var_item_flags,
                              parameter_item_weights)]) <= max_weight

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=10000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'select_items: 'f'{", ".join([x for x in answer.keys() if answer[x] == 1])}')

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

from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


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


# 単純なTSP問題
# ルート/ポイント数を設定
PLACE_NUM = 30
# 行き先名を生成
destination_names = [f'destination_{no}' for no in range(PLACE_NUM)]
# ポイント名を生成
place_names = [f'P{no}' for no in range(PLACE_NUM)]
# 距離を生成
distances = generate_distances(place_names)

# ルート変数を定義
destinations = [CategoryVariable(name=destination_name, categories=place_names)
                for destination_name in destination_names]

# 問題を設定
problem = Problem(is_max_problem=False)


# 目的関数として、距離を計算する関数を定義
def calc_distance(var_destinations, para_distances):
    return sum([para_distances[(x, y)] for x, y
                in zip(['start'] + var_destinations, var_destinations + ['start'])])


# 目的関数を定義
problem += Objective(objective=calc_distance,
                     args_map={'var_destinations': destinations, 'para_distances': distances})

# 必ず1度以上、全てのポイントに到達する制約式を追加
for place_name in place_names:
    problem += sum([(destination == place_name)
                   for destination in destinations]) >= 1

# 最適化実施
solver = OptSolver(round_times=4, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=50000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
root = ['start'] + [answer[root] for root in destination_names] + ['start']
print(f'root: {" -> ".join(root)}')

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

import numpy as np

from codableopt import Problem, Objective, IntVariable, OptSolver, \
    PenaltyAdjustmentMethod


# 顧客数、CM数を設定
CUSTOMER_NUM = 1000
CM_NUM = 100
SELECTED_CM_LIMIT = 10
# 顧客がCMを見る確率を生成
view_rates = np.random.rand(CUSTOMER_NUM, CM_NUM) / 10 / SELECTED_CM_LIMIT

# CMの放送有無の変数を定義
cm_times = [IntVariable(name=f'cm_{no}', lower=0, upper=1) for no in range(CM_NUM)]

# 問題を設定
problem = Problem(is_max_problem=True)


# 目的関数として、CMを1度でも見る確率を計算
def calculate_view_rate_sum(var_cm_times, para_non_view_rates):
    selected_cm_noes = \
        [cm_no for cm_no, var_cm_time in enumerate(var_cm_times) if var_cm_time == 1]
    view_rate_per_customers = np.ones(para_non_view_rates.shape[0]) \
        - np.prod(para_non_view_rates[:, selected_cm_noes], axis=1)
    return np.sum(view_rate_per_customers)


# 目的関数を定義
problem += Objective(objective=calculate_view_rate_sum,
                     args_map={'var_cm_times': cm_times,
                               'para_non_view_rates': np.ones(view_rates.shape) - view_rates})

# CMの選択数の制約式を追加
problem += sum(cm_times) <= SELECTED_CM_LIMIT

print(problem)

# 最適化実施
solver = OptSolver(round_times=2, debug=True, debug_unit_step=1000)
method = PenaltyAdjustmentMethod(steps=100000)
answer, is_feasible = solver.solve(problem, method, n_jobs=-1)

print(f'answer_is_feasible:{is_feasible}')
print(f'selected cm: {[cm_name for cm_name in answer.keys() if answer[cm_name] == 1]}')

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

from typing import List

import pandas as pd

from sample.usage.problem.matching_problem_generator import MatchingProblemGenerator
from codableopt import Problem, Objective, CategoryVariable, OptSolver, PenaltyAdjustmentMethod


# 最適化問題定義
print('generate Problem')
matching = MatchingProblemGenerator.generate(customer_num=1000, item_num=20)

print('start Optimization')
# 変数を定義
selected_items = \
    [CategoryVariable(name=f'item_for_{customer.name}', categories=matching.item_names)
     for customer in matching.customers]
selected_coupons = \
    [CategoryVariable(name=f'coupon_for_{customer.name}', categories=matching.coupon_names)
     for customer in matching.customers]


# 利益の期待値を計算する関数、目的関数に利用
def calculate_benefit(
        var_selected_items: List[str],
        var_selected_coupons: List[str],
        para_customer_features_df: pd.DataFrame,
        para_item_features_df: pd.DataFrame,
        para_coupon_features_df: pd.DataFrame,
        para_buy_rate_model):

    features_df = pd.concat([
        para_customer_features_df.reset_index(drop=True),
        para_item_features_df.loc[var_selected_items, :].reset_index(drop=True),
        para_coupon_features_df.loc[var_selected_coupons, :].reset_index(drop=True)
    ], axis=1)

    # 目的関数内で機械学習モデルを利用
    buy_rate = \
        [x[1] for x in para_buy_rate_model.predict_proba(features_df.drop(columns='item_cost'))]

    return sum(
        buy_rate *
        (features_df['item_price'] - features_df['item_cost'] - features_df['coupon_down_price']))


# 問題を設定
problem = Problem(is_max_problem=True)

# 目的関数を定義
problem += Objective(objective=calculate_benefit,
                     args_map={
                         'var_selected_items': selected_items,
                         'var_selected_coupons': selected_coupons,
                         'para_customer_features_df': matching.customer_features_df,
                         'para_item_features_df': matching.item_features_df,
                         'para_coupon_features_df': matching.coupon_features_df,
                         'para_buy_rate_model': matching.buy_rate_model
                     })

# 制約式を定義
for item in matching.items:
    # 必ず1人以上のカスタマーにアイテムを表示する制約式を設定
    problem += sum([(x == item.name) for x in selected_items]) >= 1
    # 同じアイテムの最大表示人数を制限する制約式を設定
    problem += sum([(x == item.name) for x in selected_items]) <= matching.max_display_num_per_item

for coupon in matching.coupons:
    # クーポンの最大発行数の制約式を設定
    problem += \
        sum([(x == coupon.name) for x in selected_coupons]) <= matching.max_display_num_per_coupon


# 最適化実施
print('start solve')
solver = OptSolver(round_times=1, debug=True, debug_unit_step=1000,
                   num_to_tune_penalty=100, num_to_select_init_answer=1)
method = PenaltyAdjustmentMethod(steps=40000)
answer, is_feasible = solver.solve(problem, method)

print(f'answer_is_feasible:{is_feasible}')
print(answer)

