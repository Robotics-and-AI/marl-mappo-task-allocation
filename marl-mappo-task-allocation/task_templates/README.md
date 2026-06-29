# Task templates

`task_templates.yaml` is the natural-language task-template dataset used by `NLTaskGenerator` in `task_generator.py`.

Each scenario contains parameter ranges, train templates, and eval templates. The top-level `requirements` mapping gives the ground-truth requirement vector for each scenario in this order:

```text
[mobility_class, manipulation_class, payload_class]
```

The default generator path is:

```text
task_templates/task_templates.yaml
```

You can also load a custom template dataset in Python:

```python
from task_generator import NLTaskGenerator

generator = NLTaskGenerator(templates_path="path/to/custom_task_templates.yaml")
task = generator.generate_task("scenario1", train_mode=True)
```
