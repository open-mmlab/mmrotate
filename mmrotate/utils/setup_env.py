# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmrotate into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmrotate default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmrotate`, anmmrotate all registries will build modules from mmrotate's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmrotate.datasets  # noqa: F401,F403
    import mmrotate.evaluation  # noqa: F401,F403
    import mmrotate.models  # noqa: F401,F403
    import mmrotate.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmrotate')
        if never_created:
            DefaultScope.get_instance('mmrotate', scope_name='mmrotate')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmrotate':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmrotate", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmrotate". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmrotate-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmrotate')
