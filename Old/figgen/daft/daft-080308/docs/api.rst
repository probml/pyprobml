.. _api:

API
===

.. module:: daft

The PGM Object
--------------

All daft scripts will start with the creation of a :class:`PGM` object. This
object contains a list of :class:`Node` objects and :class:`Edge` objects
connecting them. You can also specify rendering parameters and other default
parameters when you initialize your :class:`PGM`.

.. autoclass:: PGM
   :inherited-members:


Nodes
-----

.. autoclass:: Node
   :inherited-members:


Edges
-----

.. autoclass:: Edge
   :inherited-members:


Plates
------

.. autoclass:: Plate
   :inherited-members:


The Rendering Context
---------------------

.. autoclass:: _rendering_context
   :inherited-members:
