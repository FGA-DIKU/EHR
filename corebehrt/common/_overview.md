

### azure.py
Saved for later, should be revamped with all azure stuff

### config.py
Is this possible to simplify?
-> switch to hydra?/ Use CLI instead of config

### initalize.py
Classes: Initalizer, ModelManager
 - How much are these needed? Takes care of some things but can be improved with torch.lightning

Functional: Basically all

Notes: Avoid config

### loader.py
classes: FeaturesLoader, ModelLoader
 - Possibly combine to a single Loader?

Functional: Basically all

Notes: Revamp to be less dependent on specific file extensions

### logger.py
Don't understand? Is this really needed? -> useful but there might be other ways?

### saver.py
Classes: Saver

Functional: Basically all
 - Except they should be simplified - make a general function (e.g. just object, name args)

Notes: Make less dependent on specific naming conventions and inputs

### setup.py
classes: DirectoryPreparer

functional: Basically all

Notes: Might be simplified via. Azure revamp

### utils.py
classes: Data
 - Should we move away from this class? -> I would keep it, otherwise one needs to pass multiple objects which belong together

Functional: basically all

Notes:
 - Do we want the Data calss?
 - Avoid config