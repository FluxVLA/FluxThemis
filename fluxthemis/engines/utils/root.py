from mmengine.registry import Registry

ENVIRONMENTS = Registry('environments', scope='fluxthemis')
EVALUATORS = Registry('evaluators', scope='fluxthemis')
POLICY_CLIENTS = Registry('policy_clients', scope='fluxthemis')
EVALUATION_REPORTERS = Registry('evaluation_reporters', scope='fluxthemis')
