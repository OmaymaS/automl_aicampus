from typing import List, Optional
from enum import IntEnum
import numpy as np
import logging
import random


# The two following classes just make it convenient to select which mutation/recombination/selectoin to use with EA
class Recombination(IntEnum):
    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    INTERMEDIATE = 1  # intermediate recombination


class Mutation(IntEnum):
    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    GAUSSIAN = 1  # Gaussian mutation


class ParentSelection(IntEnum):
    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


class Member:
    """
    Class to simplify member handling.
    """

    def __init__(self, initial_x: np.ndarray, target_function: callable, bounds: List[float],
                 mutation: Mutation, recombination: Recombination,
                 sigma: Optional[float] = None, recom_prob: Optional[float] = None) -> None:
        """
        Init
        :param initial_x: Initial coordinate of the member
        :param target_function: The target function that determines the fitness value
        :param bounds: Allowed bounds. For simplicities sake we assume that all elements in initial_x have the same
                       bounds -> bounds[0] lower bound && bounds[1] upper bounds
        :param mutation: hyperparameter that determines which mutation type use
        :param recombination: hyperparameter that determines which recombination type to use
        :param sigma: Optional hyperparameter that is only active if mutation is gaussian
        :param recom_prob: Optional hyperparameter that is only active if recombination is uniform
        """
        
        self._x = np.array(initial_x, dtype=np.float32)
        ## initial_x.astype(float)  # astype is crucial here. Otherwise numpy might cast everything to int
        self._f = target_function
        self.__bounds = bounds
        self._age = 0  # basically indicates how many offspring were generated from this member
        self._mutation = mutation
        self._recombination = recombination
        self._x_changed = True
        self._fit = None
        self._sigma = sigma
        self._recom_prob = recom_prob
        self.logger = logging.getLogger(self.__class__.__name__)

    @property  # fitness can only be queried never set
    def fitness(self):
        if self._x_changed:  # Only if the x_coordinate has changed we need to evaluate the fitness.
            self._x_changed = False
            self._fit = self._f(self._x)
        return self._fit  # otherwise we can return the cached value

    @property  # properties let us easily handle getting and setting without exposing our private variables
    def x_coordinate(self):
        return self._x

    @x_coordinate.setter
    def x_coordinate(self, value):
        assert np.all((self.__bounds[0] <= value) & (value <= self.__bounds[1])), 'Member out of bounds'
        self._x_changed = True
        self._x = value

    def mutate(self):
        """
        Mutation which creates a new offspring
        :return: new member who is based on this member
        """
        new_x = self.x_coordinate.copy()
        self.logger.debug('new point before mutation:')
        self.logger.debug(new_x)
        if self._mutation == Mutation.UNIFORM:
            new_x = np.random.uniform(low=self.__bounds[0], high=self.__bounds[1])
        elif self._mutation == Mutation.GAUSSIAN:
            assert self._sigma, 'Sigma has to be set when gaussian mutation is used'
            new_x = new_x+random.gauss(0,self._sigma)
            new_x[new_x < self.__bounds[0]] = self.__bounds[0] ## apply min limit (OR subtract?!)
            new_x[new_x > self.__bounds[1]] = self.__bounds[1] ## apply max limit (OR subtract?!)?
                
        elif self._mutation != Mutation.NONE:
            pass
        self.logger.debug('new point after mutation:')
        self.logger.debug(new_x)
        child = Member(new_x, self._f, self.__bounds, self._mutation, self._recombination,
                       self._sigma, self._recom_prob)
        self._age += 1
        return child

    def recombine(self, partner):
        """
        Recombination of this member with a partner
        :param partner: Member
        :return: new offspring based on this member and partner
        """
        if self._recombination == Recombination.INTERMEDIATE:
            # TODO
            raise NotImplementedError
        elif self._recombination == Recombination.UNIFORM:
            assert self._recom_prob is not None, \
                'for this recombination type you have to specify the recombination probability'
            # TODO
            raise NotImplementedError
        elif self._recombination == Recombination.NONE:
            new_x = self.x_coordinate.copy()  # copy is important here to not only get a reference
        else:
            raise NotImplementedError
        self.logger.debug('new point after recombination:')
        self.logger.debug(new_x)
        child = Member(new_x, self._f, self.__bounds, self._mutation, self._recombination,
                       self._sigma, self._recom_prob)
        self._age += 1
        return child

    def __str__(self):
        """Makes the class easily printable"""
        str = "Population member: Age={}, x={}, f(x)={}".format(self._age, self.x_coordinate, self.fitness)
        return str

    def __repr__(self):
        """Will also make it printable if it is an entry in a list"""
        return self.__str__() + '\n'


class EA:
    def __init__(self, target_func: callable, population_size: int = 10, problem_dim: int = 2,
                 problem_bounds: List = [-30, 30], mutation_type: Mutation = Mutation.UNIFORM,
                 recombination_type: Recombination = Recombination.INTERMEDIATE,
                 sigma: float = 1., recom_proba: float = 0.5, selection_type: ParentSelection = ParentSelection.NEUTRAL,
                 total_number_of_function_evaluations: int = 200, children_per_step: int = 5,
                 fraction_mutation: float = .5
                 ):
        """
        Simple evolutionary algorithm
        :param target_func: callable target function we optimize
        :param population_size: int
        :param problem_dim: int
        :param problem_bounds: list[int] used to make sure population members are valid
        :param mutation_type: hyperparameter to set mutation strategy
        :param recombination_type: hyperparameter to set recombination strategy
        :param sigma: conditional hyperparameter dependent on mutation_type GAUSSIAN
        :param recom_proba: conditional hyperparameter dependent on recombination_type UNIFORM
        :param selection_type: hyperparameter to set selection strategy
        :param total_number_of_function_evaluations: maximum allowed function evaluations
        :param children_per_step: how many children to produce per step
        :param fraction_mutation: balance between sexual and asexual reproduction
        """
        assert 0 <= fraction_mutation <= 1
        assert 0 < children_per_step
        assert 0 < total_number_of_function_evaluations
        assert 0 < sigma
        assert 0 < problem_dim
        assert 0 < population_size
        # Step 1: initialize Population
        self.population = [
            Member(np.random.uniform(*problem_bounds, problem_dim),
                   target_func, problem_bounds, mutation_type, recombination_type, sigma, recom_proba
                   ) for _ in range(population_size)]
        self.population.sort(key=lambda x: x.fitness)  # sort population by fitness for easier handling downstream
        self.pop_size = population_size
        self.selection = selection_type
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initial average fitness of population: %f', self.get_average_fitness())
        self.max_func_evals = total_number_of_function_evaluations
        self._func_evals = population_size
        self.num_children = children_per_step
        self.frac_mutants = fraction_mutation
        # will store the optimization trajectory and lets you easily observe how often
        self.trajectory = [self.population[0]]
        # a new best member was generated

    def get_average_fitness(self) -> float:
        """Helper to quickly access average population fitness"""
        return np.mean(list(map(lambda x: x.fitness, self.population)))

    def select_parents(self):
        """
        Method that implements all selection mechanism.
        For ease of computation we assume that the population members are sorted according to their fitness
        :return: list of ids of selected parents.
        """
        parent_ids = []
        if self.selection == ParentSelection.NEUTRAL:
            # TODO
            raise NotImplementedError
        elif self.selection == ParentSelection.FITNESS:
            # TODO
            raise NotImplementedError
        elif self.selection == ParentSelection.TOURNAMENT:
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.logger.debug('Selected parents:')
        self.logger.debug(parent_ids)
        return parent_ids

    def step(self) -> float:
        """
        Performs one step of parent selection -> offspring creation -> survival selection
        :return: average population fittness
        """
        # Step 2: Parent selection
        parent_ids = self.select_parents()

        # Step 3: Variation / create offspring
        children = []
        for id in parent_ids:
            # TODO for each parent create exactly one offspring (use the frac_mutants) parameter to determine
            # if more recombination or mutation should be performed
            raise NotImplementedError
            self._func_evals += 1
        self.logger.debug('Children:')
        self.logger.debug(children)

        # Step 4: Survival selection
        # (\mu + \lambda)-selection i.e. combine offspring and parents in one sorted list, keep the #pop_size best
        self.population.extend(children)
        self.population.sort(key=lambda x: x.fitness)
        self.population = self.population[:self.pop_size]
        self.trajectory.append(self.population[0])
        return self.get_average_fitness()

    def optimize(self):
        """
        Simple optimization loop that stops after a predetermined number of function evaluations
        :return:
        """
        step = 1
        while self._func_evals < self.max_func_evals:
            avg_fitness = self.step()
            self.logger.info(
                'Step {:>3d} | Average fitness {:>10.7f} | Best fitness {:>10.7f} | #Func Evals: {:>4d}'.format(
                    step, avg_fitness, self.population[0].fitness, self._func_evals))
            step += 1
        return self.population[0]


if __name__ == '__main__':
    """
    Simple main to give an example of how to use the EA
    """
    from target_function import ackley

    np.random.seed(0)  # fix seed for comparisons sake
    logging.basicConfig(level=logging.INFO)
    dimensionality = 2
    max_func_evals = 500 * dimensionality
    pop_size = 20

    ea = EA(ackley, pop_size, dimensionality, selection_type=ParentSelection.TOURNAMENT,
            total_number_of_function_evaluations=max_func_evals)
    optimum = ea.optimize()
    # print(ea.trajectory)
    print(optimum)
    print('#' * 120)
    ea = EA(ackley, pop_size, dimensionality, selection_type=ParentSelection.FITNESS,
            total_number_of_function_evaluations=max_func_evals)
    optimum = ea.optimize()
    # print(ea.trajectory)
    print(optimum)
    print('#' * 120)
    ea = EA(ackley, pop_size, dimensionality, selection_type=ParentSelection.NEUTRAL,
            total_number_of_function_evaluations=max_func_evals)
    optimum = ea.optimize()
    # print(ea.trajectory)
    print(optimum)
    print('#' * 120)
