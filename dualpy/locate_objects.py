"""Searches an object for instances of a given type.



"""

import collections
import inspect
import logging
from typing import Any, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["LocatedObjectIterator"]

_default_terminal_types = [np.ndarray, float]


@dataclass
class LOTreeNode:
    """Describing inputs/outputs to functions in numeric Jacobian computation

    This is one node in a tree that decomoses an object to describe where dlarray
    objects are within an object instance.  Note that is is rather a child-centric
    implementation, describing how to find the said node within its parent.

    Attributes:
    -----------

    attribute_name: str, optional
        If set, gives the name under which this node is to be found as an attribute of
        its parent.
    key: any, optional
        If set, gives a key under which this node is to be found as an item of its
        parent
    contents : str or NJTreeNode
        Contains the node that this child represents, or a str containing one of the
        sought-after types if it is one of those.
    """

    attribute_name: str = None
    key: Any = None
    contents: Union[type, "LOTreeNode"] = None

    def get_corresponding_items(self, *sources):
        """Get from the source(s) the item that this tree node describes"""
        if self.attribute_name is not None:
            return tuple(getattr(source, self.attribute_name) for source in sources)
        if self.key is not None:
            return tuple(source[self.key] for source in sources)


class LocatedObjectIterator:
    """Allows recursive identification and rapid access to specific components of class"""

    def __init__(
        self,
        obj,
        targets: list[type, str],
        terminal_attributes=None,
        terminal_types=None,
    ):
        """Recursively searches through an object to find components of given type

        Paramters:
        ----------

        obj : any
            Object to be recursively searched

        targets : list[type, str]
            The objects to specifically identify within the object

        """
        # Store the targets information
        if isinstance(targets, type):
            targets = [targets]
        self.targets = targets
        # Store the terminal attributes information
        if terminal_attributes is None:
            terminal_attributes = []
        if isinstance(terminal_attributes, str):
            terminal_attributes = [terminal_attributes]
        self.terminal_attributes = terminal_attributes
        # Store the terminal types information
        if terminal_types is None:
            terminal_types = _default_terminal_types
        if isinstance(terminal_types, type):
            terminal_types = [terminal_types]
        self.terminal_types = terminal_types
        # Store some internal information
        self._memo = set()
        self._types = {}
        self._len = 0
        # Now go through and build the tree
        try:
            self._tree = self._search_object(obj, depth=0)
        except RecursionError:

            culprits = {
                key: value
                for key, value in sorted(self._types.items(), key=lambda item: -item[1])
            }
            raise ValueError(
                f"Input contains recursive type, likely culprits are (in reverse order): {culprits} "
            )

    def __str__(self):
        return (
            f"ObjectLocator: targets={self.targets}, "
            f"terminal_attributes={self.terminal_attributes}, "
            f"terminal_types={self.terminal_types},"
            f"contents={self._tree}"
        )

    def _object_is_target(self, obj):
        """Returns true if the objects supplied is a target"""
        return any([isinstance(obj, this_type) for this_type in self.targets])

    def _search_object(self, obj, depth):
        """Recursivelly search through object looking for target types

        Parameters:
        -----------
        a : any
            Object instance to be recurisively searched for duals.

        depth : int, optinonal default 0
            Used for tracking/reporting recursion depth

        Returns:
        --------

        result : list[NJTreeNode]
            List that constitutes the root of the recursed tree to find duals.  Irrelevant
            brances/leaves are dropped?
        """
        prefix = "--" * depth + ": "
        # If this is something we've seen before, then report it by returning None
        if id(obj) in self._memo:
            logger.debug(prefix + "Found a previously-seen item, returning None")
            return None
        # Note this object so we don't go through it again.
        self._memo.add(id(obj))
        # Note this type
        if type(obj) not in self._types:
            self._types[type(obj)] = 1
        else:
            self._types[type(obj)] += 1
        # If this is a target type then we've come to a termination, report that
        if self._object_is_target(obj):
            logger.debug(prefix + f"Found a target, returning {type(obj)}")
            self._len += 1
            return type(obj)
        # If this is one of our terminal types, then return None so it is ignored (note
        # we do this here because a type can be both terminal and target)
        if any(isinstance(obj, tt) for tt in self.terminal_types):
            logger.debug(prefix + "Skipping entry as it is a {type(obj)}")
            return None
        # If this is a collection of some kind look through its elements.  First consider
        # lists/tuples.
        branches = []
        if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str):
            logger.debug(prefix + "Is a sequence")
            for key, item in enumerate(obj):
                logger.debug(
                    prefix + f"Checking out item #{key}, which is a {type(item)}"
                )
                branch = self._search_object(item, depth=depth + 1)
                if branch:
                    branches.append(LOTreeNode(key=key, contents=branch))
            return branches
        # Now consider dicts
        if isinstance(obj, collections.abc.Mapping):
            logger.debug(prefix + "Is a mapping")
            for key, item in obj.items():
                logger.debug(prefix + f"Checking out {key}, which is a {type(item)}")
                branch = self._search_object(item, depth=depth + 1)
                if branch:
                    branches.append(LOTreeNode(key=key, contents=branch))
            return branches
        # OK, it's not a target or a collection (or to be ignored), so instead we'll go
        # through its attributes and make nodes out of them.
        for name, value in inspect.getmembers(obj):
            # Skip dunders, callables, and properties
            if (
                name.startswith("__")
                or callable(value)
                or isinstance(getattr(type(obj), name, None), property)
            ):
                logger.debug(prefix + f"Skipping {name} as dunder/callable/property")
                continue
            # Also skip any attributes we've been told to ignore
            if name in self.terminal_attributes:
                logger.debug(prefix + f"Skipping {name} as on ignored list")
                continue
            logger.debug(prefix + f"Checking out attribute {name}")
            branch = self._search_object(value, depth=depth + 1)
            if branch:
                logger.debug(prefix + f"Appending {name}, {branch}")
                branches.append(LOTreeNode(attribute_name=name, contents=branch))
        logger.debug(prefix + f"Returning {branches}")
        return branches

    def _iterate_worker(self, branch, *sources: Any):

        """Iterate over tree describing numerical Jacobian inputs/outputs

        Iterates over thee tree within self, and uses that tree to walk through as many
        objects as provided in sources in parallel (those objects are assumed to match
        the structure described in the tree).

        Arguments
        ---------

        branch : string or SOTreeNode
            Output of self._search_object, or part of that output possibly.

        sources : sequence of objects
            Objects from which to extract target objects.
        """
        if isinstance(branch, type):
            if len(sources) == 1:
                yield sources[0]
            else:
                yield sources
        else:
            for child in branch:
                elements = child.get_corresponding_items(*sources)
                yield from self._iterate_worker(child.contents, *elements)

    def __len__(self):
        return self._len

    def __call__(self, *sources):
        """Iterate over all the matching elements within a collection of sources

        Parameters:
        -----------

        *sources : tuple(Any)
            The objects to iterate over.  Assumed to match the stucture of that
            presented to ObjectLocator on creation.
        """
        yield from self._iterate_worker(self._tree, *sources)
