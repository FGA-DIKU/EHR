

### config.py
Is this possible to simplify?

### initalize.py
Classes: Initalizer, ModelManager
 - How much are these needed?

Functional: Basically all

Notes: Avoid config

### loader.py
classes: FeaturesLoader, ModelLoader
 - Possibly combine to a single Loader?

Functional: Basically all

Notes: Revamp to be less dependent on specific file extensions

### logger.py
Don't understand? Is this really needed?

### saver.py
Classes: Saver

Functional: Basically all
 - Except they should be simplified - make a general function (e.g. just object, name args)

Notes: Make less dependent on specific naming conventions and inputs

### setup.py
classes: DirectoryPreparer

functional: Basically all

### utils.py
classes: Data
 - Should we move away from this class?

Functional: basically all

Notes:
 - Do we want the Data calss?
 - Avoid config
