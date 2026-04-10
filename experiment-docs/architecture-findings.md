# Architecture Findings

## Schema generation leaks dependency-injected parameters to the model

`cloud_drive` is a runtime-injected dependency (`Depends("cloud_drive")`) that the model should never see. `src/Environments.py:466` includes it in the tool schema as a required string with the auto-generated description `"The cloud drive."` — giving the model no guidance on what value to provide. Haiku defaulted to echoing the parameter name (`"cloud_drive": "cloud_drive"`). The search still succeeded because the runtime injects the real value regardless, but this is a bug in schema construction. The SHADE Arena runtime in `functions_runtime.py` correctly filters these out; `Environments.py` should do the same.
