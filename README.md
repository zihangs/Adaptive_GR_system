# Implementations of continuous goal recognition systems

The continuous GR systems are implemented based on a standard process mining-based "single-shot" GR system (can be accessed [here](https://github.com/zihangs/GR_system)). ``recognizer.jar`` is the complied tool of the standard GR system.

The continuous systems have an additional module, relearn controller, which is compiled as ``controller.jar`` (the source code of the controller is in ``RelearnController/``).

The relearn controller also controls a miner to discover process models from recently collected traces. The miner is compiled as ``miner.jar``, the source code can be accessed [here](https://github.com/zihangs/miner).

The script for running the continuous GR systems is ``relearn_grsystems.py``.

```sh
python3 relearn_grsystems.py
```

Before running the above command, the parameters in the script need to be configured:

1. Input data (lines 382-382), need to specify the type of drift.
2. Output directory to store the results for each drift (line 393).
3. The percentage of observation (line 385).
4. The option of continuous GR system (line 391): standard, open-loop, closed-loop with averaged accuracy, closed-loop with trend.



Still developing: the relearn decision based on reinforcement learning ``rlsb3_system.py``.

```sh
python3 rlsb3_system.py
```



Dataset: ``experimental_data``

Dataset for developing: ``data_rl``
