from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import highspy
import numpy as np

STATUS_RU = {
    "Optimal": ("optimal", "доказан глобальный оптимум: допустимость и двойственная допустимость выполнены"),
    "Infeasible": ("infeasible", "ограничения несовместны: плана, закрывающего все требования, не существует"),
    "Unbounded": ("unbounded", "цель неограниченна: в модели не хватает ограничения"),
    "Unbounded or infeasible": ("infeasible_or_unbounded", "presolve не различил; нужен повтор без presolve"),
    "Time limit reached": ("time_limit", "лимит времени: оптимальность не доказана"),
}


@dataclass
class Result:
    status: str
    meaning: str
    objective: Optional[float]
    x: Optional[np.ndarray]
    col_dual: Optional[np.ndarray]
    row_dual: Optional[np.ndarray]
    iterations: int
    highs: highspy.Highs


def _lp_object(c, A, b, lb, ub, integrality=None):
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(c, float)
    lp.col_lower_ = np.asarray(lb, float)
    lp.col_upper_ = np.asarray(ub, float)
    lp.row_lower_ = np.full(A.shape[0], -highspy.kHighsInf)
    lp.row_upper_ = np.asarray(b, float)
    start, index, value = [0], [], []
    for col in A.T:
        nz = np.nonzero(col)[0]
        index.extend(nz.tolist())
        value.extend(col[nz].tolist())
        start.append(len(index))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array(start)
    lp.a_matrix_.index_ = np.array(index)
    lp.a_matrix_.value_ = np.array(value, float)
    if integrality is not None:
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in integrality]
    return lp


def solve(c, A, b, ub, lb=None, integrality=None, log: Optional[Path] = None,
          solver: str = "simplex", presolve: str = "on") -> Result:
    lb = np.zeros(len(c)) if lb is None else lb
    h = highspy.Highs()
    h.setOptionValue("output_flag", log is not None)
    h.setOptionValue("log_to_console", False)
    if log is not None:
        log.unlink(missing_ok=True)
        h.setOptionValue("log_file", str(log))
    h.setOptionValue("presolve", presolve)
    if integrality is None:
        h.setOptionValue("solver", solver)
    else:
        h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("random_seed", 10101)
    h.passModel(_lp_object(c, A, b, lb, ub, integrality))
    h.run()
    raw = h.modelStatusToString(h.getModelStatus())
    if raw == "Unbounded or infeasible" and presolve == "on":
        return solve(c, A, b, ub, lb, integrality, log, solver, "off")
    status, meaning = STATUS_RU.get(raw, (raw, "см. лог солвера"))
    info = h.getInfo()
    if status != "optimal":
        return Result(status, meaning, None, None, None, None, info.simplex_iteration_count, h)
    sol = h.getSolution()
    return Result(status, meaning, info.objective_function_value, np.array(sol.col_value),
                  np.array(sol.col_dual), np.array(sol.row_dual) if integrality is None else None,
                  info.simplex_iteration_count, h)


