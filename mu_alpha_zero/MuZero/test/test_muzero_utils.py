import pytest


def test_add_actions_to_obs():
    from mu_alpha_zero.MuZero.utils import add_actions_to_obs
    import torch as th

    # test handles non tensor inputs
    inpt = [1, 2, 3]
    actions = [4, 5, 6]
    with pytest.raises(AttributeError):
        add_actions_to_obs(inpt, actions)

    with pytest.raises(AttributeError):
        add_actions_to_obs(th.tensor(inpt), actions)

    with pytest.raises(AttributeError):
        add_actions_to_obs(inpt, th.tensor(actions))

    # test handles wrong dimensions
    inpt = th.rand((10, 10))
    actions = th.rand((10, 10))
    with pytest.raises(RuntimeError):
        add_actions_to_obs(inpt, actions, dim=2)
    with pytest.raises(RuntimeError):
        add_actions_to_obs(inpt, actions, dim=-3)

    # test handles correct inputs
    inpt = th.rand((10, 10))
    actions = th.rand((10, 1))
    out = add_actions_to_obs(inpt, actions, dim=1)
    assert out.shape == (10, 11)

def test_match_action_with_obs():
    from mu_alpha_zero.MuZero.utils import match_action_with_obs
    import torch as th
    from mu_alpha_zero.config import MuZeroConfig

    # test handles non tensor inputs
    inpt = [1, 2, 3]
    action = 4
    config = MuZeroConfig()
    with pytest.raises(AttributeError):
        match_action_with_obs(inpt, action, config)

    #test handles non 3D tensor
    inpt = th.rand((10,10,10,10))
    action = 4
    with pytest.raises(RuntimeError):
        match_action_with_obs(inpt, action, config)


