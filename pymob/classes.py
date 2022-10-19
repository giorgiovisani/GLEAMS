import copy
import dataclasses
from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field
from tqdm import tqdm

class Node(object):
    def __init__(self, id, weights, split_var=None, terminal=False, parent_split=None, ssplit=None, prediction=0,
                 left_child=None, right_child=None, domain=None, score=None, leaf_points=None, st_devs=None):
        self.id = id
        self.weights = weights
        self.split_var = split_var
        self.terminal = terminal
        self.parent_split = parent_split
        self.ssplit = ssplit
        self.prediction = prediction
        self.left_child = left_child
        self.right_child = right_child
        self.n_samples = float(sum(weights))
        self.regression = None
        self.leaf_points = leaf_points
        self.domain = domain
        self.score = score
        self.st_devs = st_devs

    @property
    def variables(self):
        return self.domain.get_variables()

    def __hash__(self):
        x = (self.id,
             self.weights,
             self.split_var,
             self.terminal,
             self.parent_split,
             self.ssplit,
             self.prediction,
             self.left_child,
             self.right_child,
             self.n_samples,
             self.regression.coef_.tostring(),
             self.regression.intercept_.tostring())

        return hash(x)

@dataclass(init=True, frozen=True)
class Data:
    minsplit: int
    verbose: bool
    stopping_value: float
    method: str
    aggregation_function: str
    max_outliers: int
    domain_dict: dict
    ml_continuous: bool
    quantile: float
    bb_quantile: float
    stopping_criterion: str

class StoppingCriterion(ABC):
    def __init__(self, minsplit, verbose):
        self.minsplit = minsplit
        self.verbose = verbose

    def pre_fit(self, weights, statistic):
        pass

    def post_fit(self,statistic):
        pass

    def stopping_rule_minsplit(self, weights):
        """ Check if condition: num_units < 2*minsplit"""
        if sum(weights) < 2 * self.minsplit:
            self.verbose and print(f"Too few units ({sum(weights)}): no split possible\n")
            return True
        return False


class R2StoppingCriterion(StoppingCriterion):
    def __init__(self, stopping_value,*args, **kwargs):
        self.stopping_value = stopping_value
        super().__init__(*args, **kwargs)


    def stopping_rule_R2(self, statistic):
        """Check Stopping conditions for R2 stopping criterion:
        - if the required stopping value for the given statistic is reached in the node
        """
        # check if stopping condition on statistic value is reached
        if statistic >= self.stopping_value:
            self.verbose and print(
                f"R2 value of the leaf: {statistic:.2f} Stop here!\n")
            return True
        return False

    def pre_fit(self, weights,statistic):

        cond1 = self.stopping_rule_minsplit(weights=weights)
        cond2 = self.stopping_rule_R2(statistic=statistic)
        return cond1 or cond2

    def post_fit(self, statistic):
        pass


class BBStoppingCriterion(StoppingCriterion):
    def __init__(self, quantile, bb_quantile,*args, **kwargs):
        self.quantile = quantile
        self.bb_quantile = bb_quantile
        super().__init__(*args, **kwargs)

    def stopping_rule_BB(self, statistic):
        """Check Stopping conditions for R2 stopping criterion:
        - if the required stopping value for the given statistic is reached in the node
        """
        # check if stopping condition on statistic value is reached
        if statistic <= self.bb_quantile:
            self.verbose and print(
                f"R2 value of the leaf: {statistic:.2f} Stop here!\n")
            return True
        return False

    def pre_fit(self, weights, statistic):
        cond1 = self.stopping_rule_minsplit(weights=weights)
        return cond1

    def post_fit(self,statistic):
        cond2 = self.stopping_rule_BB(statistic=statistic)
        return cond2

class OrderedSplit():
    def __init__(self, variable_id, split_point=None, ordered=True):
        self.variable_id = variable_id
        self.ordered = ordered
        self.split_point = split_point

    def __repr__(self):
        return f"{self.variable_id=} {self.split_point=}"


class RealInterval:
    """
    Represents a real interval I, open or closed.

    Used to define the variables domains in the Mob model.
    """

    def __init__(self, bounds: tuple, included: tuple):
        """

        :param bounds: bounds of the interval in tuple form. e.g. (0, 1)
        :param included: tells if the bounds are included e.g for [0, 1) is (True, False)
        """
        well_formed = bounds[1] >= bounds[0]
        self.bounds = bounds if well_formed else (bounds[1], bounds[0])
        self.included = included if well_formed else (included[1], included[0])

    def contains(self, x: float, global_domain=None):
        """
        Check if a float x is contained in the interval
        :param x: a float number to check
        :param global_domain: global domain (taken from mob) of the specified variable
        :return: True or false
        """

        def greater(a, b): return a >= b if self.included[0] else a > b

        def less(a, b): return a <= b if self.included[1] else a < b

        # if global domain is passed, check whether the current domain is on the border.
        # If so, consider included also x values outside the border
        if global_domain:
            global_min, global_max = global_domain
            max_bound = math.inf if global_max == self.bounds[1] else self.bounds[1]
            min_bound = -math.inf if global_min == self.bounds[0] else self.bounds[0]
        else:
            max_bound = self.bounds[1]
            min_bound = self.bounds[0]

        res = greater(x, min_bound) and less(x, max_bound)

        return res

    def split_at(self, split: float):
        """
        Split the interval without any criteria for included
        :param split:
        :return:
        """
        if not self.contains(split):
            raise Exception(f"The split point {split} is not present in the domain")

        bounds_sx = (self.bounds[0], split)
        bounds_dx = (split, self.bounds[1])

        return bounds_sx, bounds_dx

    def perfect_split(self, split):
        """
        Perform a split using the rule:
        on the left node we put open ')' the right bound.
        The rest remains unchanged e.g.

                       [ ]
                [ )          [ ]
            [ )   [ )     [ )   [ ]

        returns left and right interval. The original interval remains unchanged
        """
        bounds_sx, bounds_dx = self.split_at(split)
        included_sx = (self.included[0], False)
        included_dx = (True, self.included[1])

        interval_sx = RealInterval(bounds_sx, included_sx)
        interval_dx = RealInterval(bounds_dx, included_dx)

        return interval_sx, interval_dx

    def open_split(self, split):
        """
        Perform a split using the rule:
        always open interval

                       [ ]
                ( )          ( )
            ( )   ( )     ( )   ( )

        returns left and right interval. The original interval remains unchanged
        """
        bounds_sx, bounds_dx = self.split_at(split)
        included_sx = (False, False)
        included_dx = (False, False)

        interval_sx = RealInterval(bounds_sx, included_sx)
        interval_dx = RealInterval(bounds_dx, included_dx)

        return interval_sx, interval_dx

    @property
    def width(self):
        return self.bounds[1] - self.bounds[0]

    # dictionary/array like access
    def __getitem__(self, item):
        return self.bounds[item]

    def __repr__(self):
        left_par = lambda included: "[" if included else "("
        right_par = lambda included: "]" if included else ")"

        return f"{left_par(included=self.included[0])}{self.bounds[0]}, {self.bounds[1]}{right_par(included=self.included[1])}"

    def __eq__(self, other):
        if isinstance(other, RealInterval):
            return self.bounds == other.bounds and self.included == other.included
        else:
            return False


class AbstractDomain(ABC):
    """
    Abstract class for a generic domain
    """

    @abstractmethod
    def contains(self, x) -> bool:
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def set_all(self, domains):
        pass


class RealDomain(AbstractDomain, ABC):
    """
    Class to represent multi-variables continuous real domains,
    where usually each variable domain is represented as a RealInterval.
    """

    def __init__(self, domains: dict):
        """
        domains: dict = {"x0": RealInterval0,
                         "x1": RealInterval1,
                         ...}
        """
        self.domains = domains

    def contains_single_var(self, x: float, var: str, global_domain=None) -> bool:
        """
        Checks if the specific domain for var contains x

        :param x: the value
        :param var: the variable
        :param global_domain: global domain (taken from mob) of the specified variable
        :return: boolean value (True, False) stating whether the x point belongs to the domain of the given variable
                (if global_domain is passed, the point returns True when it is outside the specific domain
                but the domain is at the boundary of the global domain)
        """
        return self.domains[var].contains(x, global_domain)

    def contains(self, x) -> bool:
        """
        Checks if a point x is contained in a domain
        :param x: the point
        :return: True/False
        """
        vars_in_bounds = (self.contains_single_var(x[var], var) for var in self.domains)
        return all(vars_in_bounds)

    def contains_without_var(self, x, var, global_domain=None) -> bool:
        """
        Checks if a point x is contained in the domain object.
        Additionally, we may pass a global_domain, to understand whether the actual domain lies in a boundary
        of the global one, and consequently consider included also an x point outside the global boundary

        :param x: selected point
        :param var: variable to be not considered ( it is usually the variable on which we want to slide over
                    to retrieve all the nodes for the single variable what-if scenario
        :param global_domain: global domain of the entire mob algorithm. It is a dict as {var_id, var_domain}
        :return: boolean value (True, False) stating whether the x point belongs to the given domain
                (not considering the boundaries for the variable var)
        """

        vars = [v for v in self.domains if v != var]
        vars_in_bounds = list((self.contains_single_var(x[variable], variable, global_domain[variable]) for variable in vars))
        return all(vars_in_bounds)

    # redundant
    def get_all(self):
        return self.domains

    def get(self, var):
        return self.domains[var]

    def get_variables(self):
        return list(self.domains.keys())

    # LEGACY
    def keys(self):
        return self.get_variables()

    def set_all(self, domains):
        self.domains = domains

    def set(self, var, interval):
        self.domains[var] = interval

    def copy(self):
        return RealDomain(copy.copy(self.domains))

    def split(self, split_value, var):
        """
        Perform a perfect split for a Real domain
        :param split_value: split value
        :param var: variable for split
        :return:
        """
        res_sx = self.copy()
        res_dx = self.copy()

        interval = self.domains[var]
        interval_sx, interval_dx = interval.perfect_split(split_value)

        res_sx.set(var, interval_sx)
        res_dx.set(var, interval_dx)

        return res_sx, res_dx

    def open_split(self, split_value, var):
        """
        Perform a perfect split for a Real domain
        :param split_value: split value
        :param var: variable for split
        :return:
        """
        res_sx = self.copy()
        res_dx = self.copy()

        interval = self.domains[var]
        interval_sx, interval_dx = interval.open_split(split_value)

        res_sx.set(var, interval_sx)
        res_dx.set(var, interval_dx)

        return res_sx, res_dx

    def remove(self, var):
        del self.domains[var]

    def insert(self, var: str, interval: RealInterval):
        self.domains[var] = interval

    # dictionary/array like access
    def __getitem__(self, item) -> RealInterval:
        return self.domains[item]

    def __repr__(self):
        repr = ""
        # repr += "RealDomain: \n"
        for var, interval in self.domains.items():
            repr += f"{var}: "
            repr += str(interval)
            repr += " | "

        return repr

    def __eq__(self, other):
        if isinstance(other, RealDomain):
            return all(other[k] == v for k, v in self.domains.items())
        else:
            return False

    def to_str(self):
        return {x: str(val) for x, val in self.domains.items()}



if __name__ == "__main__":
    def main_domain():
        i0 = RealInterval(bounds=(0, 1), included=(True, True))
        i1 = RealInterval(bounds=(1, 2), included=(True, False))

        d = {"x0": i0, "x1": i1}

        dom = RealDomain(d)

        # test
        print("1" + "*" * 40)
        print(i0)
        print(i1)
        print("")

        print(f"{i0} contains 0? {i0.contains(0)}")  # true
        print(f"{i0} contains 1? {i0.contains(1)}")  # true
        print(f"{i0} contains 2? {i0.contains(2)}")  # false
        print(f"{i1} contains 1? {i1.contains(1)}")  # true
        print(f"{i1} contains 2? {i1.contains(2)}")  # false
        print("")

        # split0
        sx, dx = i0.perfect_split(0.5)
        print("2" + "*" * 40)
        print(f"Split {i0=} in 0.5.")
        print(f"{sx=}")
        print(f"{dx=}")
        print("")

        # split1
        dx_1, dx_2 = dx.perfect_split(0.7)
        print("3" + "*" * 40)
        print(f"Split {dx=} from before in 0.7.")
        print(f"{dx_1=}")
        print(f"{dx_2=}")
        print("")

        # realdomain split
        d0, d1 = dom.split(0.5, "x0")
        print("4" + "*" * 40)
        print("DOM")
        print(dom)
        print("Split DOM in x0=0.5")
        print("D0")
        print(d0)
        print("D1")
        print(d1)
        print("")

        # realdomain contains
        # meglio prevedere classe point?
        print("5" + "*" * 40)
        print("DOM")
        print(dom)
        p = ({"x0": 0.5, "x1": 1})
        # print(f"DOM contains {p.coordinates}? {dom.contains(p.coordinates)}")
        print(f"DOM contains {p}? {dom.contains(p)}")  # true
        p = ({"x0": 1, "x1": 1})
        print(f"DOM contains {p}? {dom.contains(p)}")  # true
        p = ({"x0": 1, "x1": 2})
        print(f"DOM contains {p}? {dom.contains(p)}")  # false

        print("")
        print("6" + "*" * 40)
        print("Test on random variables")

        import random as r

        def rand_bounds():
            a = round(r.random(), 3)
            b = round(r.random(), 3)
            return (a, b) if a < b else (b, a)

        i_s = {f"x{x}": RealInterval(bounds=(rand_bounds()), included=(bool(r.getrandbits(1)), bool(r.getrandbits(1))))
               for
               x in range(15)}
        df = RealDomain(i_s)
        print(df)

        # playing with __getitem__
        print(df["x0"].bounds)
        print(df["x0"].included)
        print(df["x0"])


    main_domain()

class ProgressBar:
    '''
    A simple wrapper to tqdm progress bar to facilitate parallelization.
    Instantiating the class creates a progress bar that can be updated by calling the update method.
    The bar needs to be explicitly closed by calling the close method.
    '''
    def __init__(self, total):
        self.total = total
        self.counter = 0
        self.pbar = tqdm(total=total)
        
    def update(self, delta):
        self.counter += delta
        self.pbar.update(delta)
    
    def close(self):
        self.pbar.close()