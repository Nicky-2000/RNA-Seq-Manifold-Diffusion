rna-diffusion/
  pyproject.toml or setup.cfg / setup.py  (later)
  README.md
  configs/
    baseline_gaussian.yaml
    manifold_placeholder.yaml
  src/
    rna_diffusion/
      __init__.py
      data/
        __init__.py
        loaders.py
        preprocess.py
      manifold/
        __init__.py
        base.py
        identity.py
        noisers.py
      diffusion/
        __init__.py
        base.py
        networks.py
      metrics/
        __init__.py
        basic.py
      utils/
        __init__.py
        config.py
        logging.py
      experiments/
        __init__.py
        train_baseline.py
        train_manifold.py
        eval_compare.py
      infra/
        gcp/
            README.md
            create_vm.sh
            startup.sh
  tests/
    test_data.py
    test_manifold.py
    test_diffusion.py
    test_metrics.py



General Pipeline: 

data -> manifold -> diffusion -> metrics
