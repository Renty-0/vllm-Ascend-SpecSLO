"""Post-return device checks at existing untimed common stages, no Native edit."""
def require_finite(label,*groups):
 import torch
 count=0
 for values in groups:
  for value in values:
   if not isinstance(value,torch.Tensor) or not value.dtype.is_floating_point:
    raise RuntimeError('V134 '+label+' requires actual floating tensor snapshots')
   if not bool(torch.isfinite(value).all().item()):
    raise RuntimeError('V134 '+label+' actual snapshot contains nonfinite values')
   count+=1
 if not count:raise RuntimeError('V134 '+label+' actual snapshot group empty')
 return dict(passed=True,actual_tensor_count=count,post_return_finite_check=True,Native_math_or_flags_changed=False)
