"""Import torch before xgboost/scikit-learn: on Windows the reverse order can trigger
`OSError: [WinError 1114] DLL initialization failed` (OpenMP runtime clash) when the full suite is collected."""
try:
    import torch  # noqa: F401
except Exception:      # torch optional for the non-sequence tests
    pass
