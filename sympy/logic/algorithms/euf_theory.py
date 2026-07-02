"""
Congruence Closure Engine for EUF

Implements:
    Nieuwenhuis & Oliveras, "Congruence Closure with Integer Offsets"
    https://link.springer.com/chapter/10.1007/978-3-540-39813-4_5

This algorithm efficiently computes the equivalence closure of a set of
ground equalities over uninterpreted functions, using union-find,
curryfication/flattening, and congruence propagation.

Terminology, variable names, and algorithm steps are consistent with those
shown in the paper, especially Section 4.

Example usage (doctest):

>>> from sympy import symbols, Function
>>> from sympy.logic.algorithms.euf_theory import EUFCongruenceClosure
>>> from sympy.assumptions.ask import Q
>>> f = Function('f')
>>> a, b, x, y = symbols('a b x y')
>>> cc = EUFCongruenceClosure([Q.eq(a, b), Q.eq(f(a), x), Q.eq(f(b), y)])
>>> cc.are_equal(x, y)
True
>>> cc.are_equal(a, x)
False

Classes
-------
    EUFCongruenceClosure: Implements the congruence closure algorithm for EUF.
"""

from collections import defaultdict, deque
from sympy.core.symbol import Symbol
from sympy.core.function import Lambda
from sympy.core.symbol import Dummy
from sympy.core.numbers import Number
from sympy.core import Basic
from sympy.utilities.iterables import numbered_symbols
from sympy.assumptions.assume import AppliedPredicate



class EUFUnhandledInput(Exception):
    """
    Raised while creating an EUFCongruenceClosure if unhandled input is present.
    """

class EUFCongruenceClosure:
    """
    Congruence closure algorithm for ground Equality with Uninterpreted Functions (EUF).

    See:
        Nieuwenhuis & Oliveras, "Congruence Closure with Integer Offsets"
        https://link.springer.com/chapter/10.1007/978-3-540-39813-4_5

    Major data structures (using algorithm's variable names):
        pending_unions:   deque, list of pairs of constants yet to be merged (PENDING).
        representative_table: dict, mapping: constant -> its class representative (REPRESENTATIVE).
        classlist: defaultdict(set), rep -> set of all elements in class (CLASSLIST).
        lookup_table: dict, maps (function, tuple of args) to a constant (LOOKUP).
        use_list: defaultdict(list), rep -> list of (func, args, result) triples (USELIST).

    Terms are transformed once and for all, before any merging (Sec. 4).
    """

    def __init__(self, equations):
        """
        Parameters
        ----------
        equations : list of Q.eq or SymPy expressions
            The ground equalities to be saturated.
        """
        # pairs of constants yet to be merged
        self.pending_unions = deque()

        # each constant has a representative (like a pointer)
        # all the constant that have the same representative are
        # in the same class
        self.representative_table = {}           # Representative[c]

        # list of all the constants in the same class.
        # Classes are list of terms currently known to be equal.
        self.classlist = defaultdict(set)        # ClassList[rep]

        # for each input *(a,b), it stores consant c s.t
        # c = *(a,b). It uses representatives of a and b.
        self.lookup_table = {}                   # Lookup_table[function, args]

        # list of input equations *(b,c) = d
        # where a is representative of b or/and c.
        self.use_list = defaultdict(list)        # UseList[rep]

        self._dummies = numbered_symbols('c', Dummy)
        self._term_to_const = {}
        self._lambda_cache = {}

        # Transform every term of the input equations first, then merge.
        for eq in equations:
            left_id = self._transform_formula(eq.lhs)
            right_id = self._transform_formula(eq.rhs)
            self.pending_unions.append((left_id, right_id))
        self._process_pending_unions()

    def _register(self, const):
        """Ensure const is in class structures as its own singleton."""
        if const not in self.representative_table:
            self.representative_table[const] = const
            self.classlist[const].add(const)

    def _new_dummy(self):
        d = next(self._dummies)
        self._register(d)
        return d

    def _transform_formula(self, expr):
        """
        Curryfy and flatten expr, assigning a constant to each unique
        subterm as in Sec. 4. Must be called before any merging.

        By flatten, we mean that this method will transform the formula into
        terms of at most depth of 2.

        Returns
        -------
        Symbol/Dummy : unique id for the term subtree.
        """
        if expr in self._term_to_const:
            return self._term_to_const[expr]

        if isinstance(expr, (Dummy, Symbol)):
            self._register(expr)
            const = expr
        elif isinstance(expr, Number) or getattr(expr, "is_Atom", False):
            const = self._new_dummy()
        elif isinstance(expr, AppliedPredicate):
            arg_ids = tuple(self._transform_formula(arg) for arg in expr.arguments)
            const = self._transform_application(expr.function, arg_ids)
        elif isinstance(expr, Lambda):
            lam = expr if len(expr.variables) == 1 else expr.curry()
            body_id = self._transform_formula(lam.expr)
            lam_key = Lambda(lam.variables[0], body_id)
            if lam_key not in self._lambda_cache:
                self._lambda_cache[lam_key] = self._new_dummy()
            const = self._lambda_cache[lam_key]
        else:
            func = expr.func
            func_id = self._transform_formula(func) if isinstance(func, Basic) else func
            arg_ids = tuple(self._transform_formula(arg) for arg in expr.args)
            const = self._transform_application(func_id, arg_ids)

        self._term_to_const[expr] = const
        return const

    def _transform_application(self, func, arg_ids):
        """Record the flat definition func(arg_ids) = d and return d."""
        key = (func, arg_ids)
        if key in self.lookup_table:
            return self.lookup_table[key]
        d = self._new_dummy()
        self.lookup_table[key] = d
        for arg_id in set(arg_ids):
            self.use_list[arg_id].append((func, arg_ids, d))
        return d

    def _const_of(self, term):
        """
        Return the constant naming a transformed term.

        Raise KeyError if the term was not part of the preprocessed input.
        """
        return self._term_to_const[term]

    def _find(self, const):
        """
        Return the unique class representative for const (with path compression).
        """
        root = const
        # Find root
        while root != self.representative_table[root]:
            root = self.representative_table[root]
        # Path compression
        while const != root:
            parent = self.representative_table[const]
            self.representative_table[const] = root
            const = parent
        return root

    def _union(self, a, b):
        rep_a, rep_b = self._find(a), self._find(b)
        if rep_a == rep_b:
            return
        # Ensure |ClassList(a)| <= |ClassList(b)|
        if len(self.classlist[rep_a]) > len(self.classlist[rep_b]):
            rep_a, rep_b = rep_b, rep_a
        # Move all members of ClassList(rep_a) into ClassList(rep_b)
        for c in list(self.classlist[rep_a]):
            self.representative_table[c] = rep_b
            self.classlist[rep_b].add(c)
        del self.classlist[rep_a]
        # For each application (func, args, term) in UseList(rep_a)
        for func, arg_ids, term in list(self.use_list.pop(rep_a, [])):
            rep_args = tuple(self._find(arg) for arg in arg_ids)
            rep_term = self._find(term)
            key = (func, rep_args)
            if key in self.lookup_table:
                other = self._find(self.lookup_table[key])
                if other != rep_term:
                    self.pending_unions.append((rep_term, other))
            self.lookup_table[key] = rep_term
            self.use_list[rep_b].append((func, arg_ids, term))

    def _process_pending_unions(self):
        """
        Saturates pending_unions queue (Main loop, Paper Section 4).
        """
        while self.pending_unions:
            self._union(*self.pending_unions.popleft())

    def merge(self, lhs, rhs):
        """
        Merge the classes of two already-transformed terms and propagate
        closure.
        Raises KeyError if a term was not transformed.

        Examples
        --------
        >>> from sympy import symbols
        >>> from sympy.assumptions.ask import Q
        >>> from sympy.logic.algorithms.euf_theory import EUFCongruenceClosure
        >>> a, b, x, y = symbols('a b x y')
        >>> cc = EUFCongruenceClosure([Q.eq(a, x), Q.eq(b, y)])
        >>> cc.merge(a, b)
        >>> cc.are_equal(x, y)
        True
        """
        self.pending_unions.append((self._const_of(lhs), self._const_of(rhs)))
        self._process_pending_unions()

    def are_equal(self, lhs, rhs):
        """
        Query whether two terms are in the same class under the closure.
        Terms that were never transformed always return False.

        Examples
        --------
        >>> from sympy import symbols, Function
        >>> from sympy.logic.algorithms.euf_theory import EUFCongruenceClosure
        >>> from sympy.assumptions.ask import Q
        >>> f = Function('f')
        >>> x, y = symbols('x y')
        >>> cc = EUFCongruenceClosure([Q.eq(x, y), Q.eq(f(x), f(y))])
        >>> cc.are_equal(x, y)
        True
        >>> cc.are_equal(f(x), f(y))
        True
        >>> cc.are_equal(x, f(x))
        False
        """
        try:
            lhs_id = self._const_of(lhs)
            rhs_id = self._const_of(rhs)
        except KeyError:
            return False
        return self._find(lhs_id) == self._find(rhs_id)
