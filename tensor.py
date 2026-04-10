import cupy as cp
import numpy as np
import weakref
import warnings
from . import _array as ap
from .dtypes import *
from collections import defaultdict

class no_grad:
    def __enter__(self):
        self.old_state = Tensor._build_graph
        Tensor._build_graph = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor._build_graph = self.old_state
        # Returning False ensures exceptions inside the block are not suppressed
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
    
class Tensor:
    
    _build_graph = True

    def __init__(self, 
                 data, 
                 requires_grad=False,
                 grad_fn=None, 
                 grad_fn_name=None,
                 device=None,
                 dtype=None):
        
        ### If our data is not an Array type, convert it ###
        ### Array handles everything regarding our basic ops ###
        ### device, dtype, and anything else numpy would do ###
        self._data = ap.Array(data=data, 
                              device=device, 
                              dtype=dtype)
        
        ### Set Autograd Variables ### 
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad_fn_name = grad_fn_name
        self.grad = None
        self._is_leaf = self.requires_grad and (self.grad_fn is None)
        self._parents = ()
        self._version = 0
        self._retain_grad = False
        self._warn_retain_grad = False

    @property
    def xp(self):
        return self._data.xp

    @property
    def data(self):
        """
        simple (view) access to data
        """
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = ap.Array(value)
        return self

    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def device(self):
        return self._data.device
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def ndim(self):
        return len(self._data.shape)
    
    @property
    def is_leaf(self):
        return self._is_leaf
    
    def __repr__(self):
        """
        Pretty printing
        """

        ### Access underlying _array for printing only ! ###
        data = self.data._array

        # Convert array to string
        data_str = self.xp.array2string(
            data,
            separator=" ",
            precision=5,
            floatmode="fixed",
            max_line_width=80
        )

        # Indent all lines after the first like PyTorch
        lines = data_str.split("\n")
        if len(lines) > 1:
            indent = " " * len("tensor(")
            data_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        # Grad / requires_grad info
        grad_info = ""
        if getattr(self, "requires_grad", False):
            if getattr(self, "grad_fn", None) is not None:
                grad_info = f", grad_fn={getattr(self, 'grad_fn_name', None)}"
            else:
                grad_info = ", requires_grad=True"

        # Device info
        device_info = ""
        if "cuda" in self.device:
            device_info = f", device={self.device}"

        # Final string
        return f"tensor({data_str}{grad_info}{device_info})"
    
    def to(self, device):
        ### use the setter of our data attribute to replace our ###
        ### existing self.data with the new one on the new device ###
        self.data = self.data.to(device)
        return self

    @classmethod
    def build_graph_enabled(cls):
        return cls._build_graph
    
    @staticmethod
    def _check_broadcast(a, b):

        ## Verify that two numpy arrays are broadcastable ###
        ## This means a and b have the same number of dimensions ###
        ## I.E (1x3) + (1x1) summation is broadcasting

        ### We only really care about this when both a and b requires gradients ###
        ### as if they dont, then either a or b are just some constant ###

        ## Numpy technically supports broadcasting even when the dimensionality ###
        ## is not the same (1 x 3) + (1, ) but we wont for simplicity! ###
        if (len(a.shape) != len(b.shape)) and (a.requires_grad and b.requires_grad):
            raise ValueError(f"Incompatible Operation between {a.shape} and {b.shape}")

    def _broadcasted_grad_accumulate(self, x_shape, x_grad):

        grad_shape = x_grad.shape

        assert len(x_shape) == len(grad_shape), "Gradient and tensor shapes must be the same length! Only different by broadcasting"

        sum_axes = [idx for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)) if x_dim == 1 and grad_dim != 1]
        if sum_axes:
            x_grad = np.sum(x_grad, axis=tuple(sum_axes), keepdims=True)

        return x_grad
    
    def retain_grad(self):
        
        if not self._warn_retain_grad:
            warnings.warn(
                "You are retaining graph, intermediate gradients may not be cleared!!"
            )
            self._warn_retain_grad = True

        ### Leaf Tensors always retain grad ###
        if self.is_leaf:
            return 

        self._retain_grad = True

    def backward(self, 
                 grad=None, 
                 retain_graph=False, 
                 time_backward=False,
                 num_ops_to_show=-1): # -1 will show all of them!
        # time_backward=True
        if retain_graph:
            if not self._warn_retain_grad:
                warnings.warn(
                    "You are retaining graph, intermediate gradients may not be cleared!!"
                )
                self._warn_retain_grad = True

        # Initialize output gradient (ones in the shape of our data)
        if grad is None:
            grad = ap.Array.ones_like(self.data, dtype=self.dtype, device=self.device)

        self.grad = grad
 
        # Build topo-order
        visited = set()
        topo_order = []

        def build_topo(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            parents = getattr(t, "_parents", ())
            if parents is None:
                parents = []
            for parent_ref in parents:
                parent = parent_ref()
                if parent is not None:
                    build_topo(parent)
            topo_order.append(t)

        build_topo(self)

        # Iterate in reverse topological order
        if time_backward:
            timings = defaultdict(float) 

        for t in reversed(topo_order):
            grad_fn = t.grad_fn
            if grad_fn is not None:
                grad_fn_name = getattr(t, "grad_fn_name", grad_fn.__class__.__name__)

                if time_backward:
                    start_gpu = cp.cuda.Event()
                    end_gpu = cp.cuda.Event()
                    start_gpu.record()

                t.grad_fn(t.grad)  # accumulate into parents
                
                if time_backward:
                    end_gpu.record()
                    end_gpu.synchronize()
                    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
                    # This will rewrite the op if it occurs multiple times, 
                    # but thats ok its just to get a general idea for the ops
                    # backward past cost
                    timings[grad_fn_name] = t_gpu

                ### Drop references immediately ###
                retain_this = getattr(t, "_retain_grad", False) or retain_graph
                
                if not retain_this:
                    # Clear backward references for GC
                    t.grad_fn = None
                    t._parents = None

                    # Non-leaf tensors don't need their grad anymore
                    if not t.is_leaf:
                        t.grad = None

        if time_backward:
            print(f"\nTop {num_ops_to_show} most time-consuming backward functions:")
            sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)[:num_ops_to_show]
            for name, total_time in sorted_timings:
                print(f"{name:<30s} {total_time:.3f} ms total")

    ###################################
    ######## BINARY OPERATIONS ########
    ###################################

    """
    Binary operations are those that happen between two different Tensors!
    Simple examples are Sum, Subtract, Mult, Div, MatMul, etc... These are all
    operations that occur between two different inputs!
    """
    def __add__(self, val):

        """
        Sum of two tensors (with accumulation for brodcasting)
        O = A + B
        dO/dA = 1
        dO/dB = 1
        """

        ### If Val is a Tensor, Then Check Devices ###
        if isinstance(val, Tensor): 

            ### Check Broadcast Shape ###
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        ### If we are just summing with an scalar ###
        ### Just cast to our dtype without any issues ###
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Use standard __add__ to actually add tensors together ###
        output = np.add(self.data, val_data)
        
        ### Define Backward Function ###
        def _add_backward(input_grad):

            if self.requires_grad:
                self_grad = self._broadcasted_grad_accumulate(self.shape, input_grad)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
                self_grad = None

            if val_requires_grad:
                val_grad = self._broadcasted_grad_accumulate(val_shape, input_grad)
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

                val_grad = None

        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_add_backward if requires_grad else None,
                        grad_fn_name="<AddBackward>" if requires_grad else None,
                        device=self.device)

        ### Set Parents ###
        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)

        return output

    def __radd__(self, val):

        """
        add is not an ordered operation, A + B is the same as B + A

        In A + B, our self is A and val is B
        When we do A + B, what is really happening is A.__add__(B). 

      
        """
        return self + val
    
    def __iadd__(self, val):
        """
        Inplace operation to enable self += val
            - prevent inplace operation on leaf tensors that require grad
            - tracks version for non-leaf tensors to see if there is a mismatch
        
      
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_self_grad_fn = getattr(self, "grad_fn", None)
        old_val_grad_fn = getattr(val, "grad_fn", None)

        ### inplace op ###
        self.data += val_data

        ### increment version (default 0 if it doesn't exist but it should always) ###
        self._version = getattr(self, "_version", 0) + 1
        
        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            def _iadd_backward(input_grad):
                # Version check on leaf tensors where we really care ###
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                ### If we arent a leaf tensor we just use our original grad fn ###
                if self.requires_grad:
                    grad_self = self._broadcasted_grad_accumulate(self.shape, input_grad)
                    
                    ### If a leaf tensor just accumulate grads like normal ###
                    if self.is_leaf or getattr(self, "_retrain_grad", False):
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self

                    ### If not a leaf tensor, just use the old grad function ###
                    elif not self.is_leaf and old_self_grad_fn is not None:
                        old_self_grad_fn(grad_self)
                    
                if val_requires_grad:
    
                    grad_val = val._broadcasted_grad_accumulate(val_shape, input_grad)
                    if val.is_leaf or getattr(val, "_retrain_grad", False):
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                        
                    elif not self.is_leaf and old_val_grad_fn is not None:
                        old_val_grad_fn(grad_val)

            self.grad_fn = _iadd_backward
            self.grad_fn_name = "<IAddBackward>"

        return self

    def __sub__(self, val):

        """
        Same as __add__ but now subtraction (with accumulation for broadcasting)
        O = A - B
        dO/dA = 1
        dO/dB = -1
        """

        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None

        output = np.subtract(self.data, val_data)
        
        ### Define Backward Function ###
        def _sub_backward(input_grad):
            if self.requires_grad:
                # self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self.shape, input_grad)
                
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

            if val_requires_grad:
                # val_grad = -input_grad
                val_grad = self._broadcasted_grad_accumulate(val_shape, -input_grad)
                
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

                val_grad = None

        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_sub_backward if requires_grad else None,
                        grad_fn_name="<SubBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)
        
        return output
    
    def __rsub__(self, val):

        """
        Subtraction is an ordered operation. Lets say we want A - B where A is self and B is val
        if A is not a tensor (i.e. an int or float), __sub__ will throw an error as it doesnt know
        how to do an operation with our own tensor.

        This will enter __rsub__ where we flip the operands where B is now self and A is val. If we want
        A - B, we need to do -1 * B + A, using our __add__. 

        There are a bunch of ways to handle these exceptions, this is just one of them!
        """

        return -1 * self + val

    def __isub__(self, val):
        """
        Inplace op to enable self -= val
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_grad_fn = getattr(self, "grad_fn", None)

        ### inplace op ###
        self.data -= val_data

        ### increment version (default 0 if it doesn't exist) ###
        self._version = getattr(self, "_version", 0) + 1

        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            def _isub_backward(input_grad):
                # Version check
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                if self.requires_grad:
                    grad_self = self._broadcasted_grad_accumulate(self.shape, input_grad)
                    
                    ### If a leaf tensor just accumulate grads like normal ###
                    if self.is_leaf:
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self

                    ### If not a leaf tensor, just use the old grad function ###
                    else:
                        old_grad_fn(grad_self)
                    
                if val_requires_grad:
                    grad_val = val._broadcasted_grad_accumulate(val_shape, -input_grad)
                    if val.is_leaf:
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                    else:
                        val.grad_fn(grad_val)

            self.grad_fn = _isub_backward
            self.grad_fn_name = "<ISubBackward>"

        return self

    def __mul__(self, val):

        """
        Element-wise multiplication of two tensors (with accumulation for broadcasting)

        O = A * B
        dO/dA = B
        do/dB = A
        """

        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
            
        output = np.multiply(self.data, val_data)

        def _mul_backward(input_grad):

            if self.requires_grad:
                self_grad = np.multiply(input_grad, val_data)
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
                self_grad = None

            if val_requires_grad:
                val_grad = np.multiply(input_grad, self.data)
                val_grad = self._broadcasted_grad_accumulate(val_shape, val_grad)
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

                val_grad = None

        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad, 
                        grad_fn=_mul_backward if requires_grad else None,
                        grad_fn_name="<MulBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)

        return output
    
    def __rmul__(self, val):
        return self * val

    def __imul__(self, val):
        """
        Inplace op to enable self *= val
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_grad_fn = getattr(self, "grad_fn", None)

        ### inplace op ###
        self.data *= val_data

        ### increment version (default 0 if it doesn't exist) ###
        self._version = getattr(self, "_version", 0) + 1

        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            def _imul_backward(input_grad):
                # Only check version for leaf tensors
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                # Gradient w.r.t. self
                if self.requires_grad:
                    grad_self = input_grad * val_data
                    grad_self = self._broadcasted_grad_accumulate(self.shape, grad_self)

                    if self.is_leaf:
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self
                    else:
                        if old_grad_fn is not None:
                            old_grad_fn(grad_self)

                # Gradient w.r.t. val
                if val_requires_grad:
                    grad_val = input_grad * self.data
                    grad_val = val._broadcasted_grad_accumulate(val_shape, grad_val)

                    if val.is_leaf:
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                    else:
                        if val.grad_fn is not None:
                            val.grad_fn(grad_val)

            self.grad_fn = _imul_backward
            self.grad_fn_name = "<IMulBackward>"

        return self

    def __neg__(self):
        return self * -1

    def __matmul__(self, val):

        ### Compute MatMul ###
        output_data = np.matmul(self.data, val.data)

        def _matmul_backward(input_grad):

            if self.requires_grad:
                grad_self = np.matmul(input_grad, val.data.swapaxes(-1, -2))
                
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

                grad_self = None

            if val.requires_grad:
                grad_val = np.matmul(self.data.swapaxes(-1, -2), input_grad)
                
                if val.grad is None:
                    val.grad = grad_val
                else:
                    val.grad += grad_val
                
                grad_val = None

        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        out = Tensor(
            output_data,
            requires_grad=requires_grad,
            grad_fn=_matmul_backward if requires_grad else None,
            grad_fn_name="<MatmulBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self, val)

        return out

   
