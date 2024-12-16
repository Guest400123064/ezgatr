# Thematic Tests
The tests under this directory are organized by the thematic of the tests. For instance, the tests under the `test_regress_with_clifford.py` are focused on verifying the outputs of EzGATr geometric algebra implementation align with the `clifford` library.

## Test Regress with Clifford
Tests implemented within [`test_regress_with_clifford.py`](./test_regress_with_clifford.py) are focused on verifying the outputs of EzGATr geometric algebra implementation align with the `clifford` library.

## Test Operator Equivariance
Tests implemented within [`test_operator_equivariance.py`](./test_operator_equivariance.py) are focused on verifying certain operators are equivariant under the sandwiching product.

## Test Gradient Flow
Tests implemented within [`test_gradient_flow.py`](./test_gradient_flow.py) are focused on verifying the gradients are correctly propagated. In addition, the constants, like the basis, should not be updated during the backward pass.
