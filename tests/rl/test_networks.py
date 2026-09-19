"""Networks: encoder selection, input handling, initialisation, actor-critic and Q heads."""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from mario_play.rl.config import NetworkConfig
from mario_play.rl.networks import (
    ActorCritic,
    GridEncoder,
    MLPEncoder,
    NatureCNN,
    QNetwork,
    build_encoder,
    layer_init,
)

IMAGE_SPACE = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
GRID_SPACE = gym.spaces.Box(-1.0, 1.0, (14, 15, 16), np.float32)
VECTOR_SPACE = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
ALL_SPACES = [
    pytest.param(IMAGE_SPACE, id="image"),
    pytest.param(GRID_SPACE, id="grid"),
    pytest.param(VECTOR_SPACE, id="vector"),
]

SMALL = NetworkConfig(hidden_size=64, mlp_hidden=[32, 32])
N_ACTIONS = 7


def make_batch(space: gym.spaces.Box, batch: int = 5, seed: int = 0) -> torch.Tensor:
    """A random batch of observations in the dtype of `space`."""
    gen = torch.Generator().manual_seed(seed)
    if space.dtype == np.uint8:
        return torch.randint(0, 256, (batch, *space.shape), generator=gen, dtype=torch.uint8)
    return torch.rand((batch, *space.shape), generator=gen) * 2.0 - 1.0


def rows_gram(weight: torch.Tensor) -> torch.Tensor:
    """Gram matrix of the rows of a (possibly convolutional) weight tensor."""
    flat = weight.detach().flatten(1)
    return flat @ flat.T


# --------------------------------------------------------------------------- #
# layer_init
# --------------------------------------------------------------------------- #


def test_layer_init_is_orthogonal_with_gain_and_constant_bias():
    torch.manual_seed(0)
    layer = layer_init(nn.Linear(64, 16), std=0.5, bias=0.25)
    assert isinstance(layer, nn.Linear)
    torch.testing.assert_close(rows_gram(layer.weight), 0.25 * torch.eye(16), atol=1e-5, rtol=0)
    assert torch.all(layer.bias == 0.25)


def test_layer_init_defaults_and_conv_and_no_bias():
    torch.manual_seed(0)
    conv = layer_init(nn.Conv2d(3, 8, 3))
    torch.testing.assert_close(rows_gram(conv.weight), 2.0 * torch.eye(8), atol=1e-5, rtol=0)
    assert torch.all(conv.bias == 0.0)
    no_bias = layer_init(nn.Linear(8, 4, bias=False), std=1.0)
    torch.testing.assert_close(rows_gram(no_bias.weight), torch.eye(4), atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- #
# encoder selection and shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("space", "expected_cls", "expected_dim"),
    [
        pytest.param(IMAGE_SPACE, NatureCNN, 512, id="image"),
        pytest.param(GRID_SPACE, GridEncoder, 512, id="grid"),
        pytest.param(VECTOR_SPACE, MLPEncoder, 64, id="vector"),
    ],
)
def test_auto_encoder_selection_and_output_shape(space, expected_cls, expected_dim):
    encoder, feature_dim = build_encoder(space, NetworkConfig())
    assert isinstance(encoder, expected_cls)
    assert feature_dim == expected_dim == encoder.feature_dim
    out = encoder(make_batch(space, batch=3))
    assert out.shape == (3, expected_dim)
    assert out.dtype == torch.float32


def test_nature_cnn_follows_the_nature_dqn_layout():
    encoder = NatureCNN((4, 84, 84), hidden_size=512)
    convs = [m for m in encoder.modules() if isinstance(m, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels, c.kernel_size, c.stride) for c in convs] == [
        (4, 32, (8, 8), (4, 4)),
        (32, 64, (4, 4), (2, 2)),
        (64, 64, (3, 3), (1, 1)),
    ]
    linears = [m for m in encoder.modules() if isinstance(m, nn.Linear)]
    assert [(fc.in_features, fc.out_features) for fc in linears] == [(64 * 7 * 7, 512)]


@pytest.mark.parametrize("shape", [(1, 36, 36), (3, 60, 64), (8, 120, 128), (2, 37, 90)])
def test_nature_cnn_handles_any_spatial_size_and_channel_count(shape):
    encoder = NatureCNN(shape, hidden_size=32)
    out = encoder(torch.zeros((2, *shape), dtype=torch.uint8))
    assert out.shape == (2, 32)


@pytest.mark.parametrize("shape", [(4, 35, 84), (4, 84, 35), (3, 8, 8)])
def test_nature_cnn_rejects_images_smaller_than_36_pixels(shape):
    with pytest.raises(ValueError, match="36"):
        NatureCNN(shape)


@pytest.mark.parametrize("shape", [(14, 15, 16), (56, 15, 16), (5, 7, 9), (3, 1, 1)])
def test_grid_encoder_handles_stacked_frames_and_other_grid_sizes(shape):
    encoder = GridEncoder(shape, hidden_size=48)
    assert encoder.feature_dim == 48
    out = encoder(torch.zeros((4, *shape)))
    assert out.shape == (4, 48)


def test_grid_encoder_keeps_spatial_information():
    # Padded convolutions must not collapse the grid: moving a single active cell changes the
    # features (a global-pooling encoder would not notice).
    torch.manual_seed(0)
    encoder = GridEncoder((1, 15, 16), hidden_size=32)
    a = torch.zeros((1, 1, 15, 16))
    b = torch.zeros((1, 1, 15, 16))
    a[0, 0, 0, 0] = 1.0
    b[0, 0, 14, 15] = 1.0
    assert not torch.allclose(encoder(a), encoder(b))


def test_mlp_encoder_uses_tanh_and_configured_widths():
    encoder = MLPEncoder((6,), hidden=[20, 10])
    linears = [m for m in encoder.modules() if isinstance(m, nn.Linear)]
    assert [(fc.in_features, fc.out_features) for fc in linears] == [(6, 20), (20, 10)]
    assert sum(isinstance(m, nn.Tanh) for m in encoder.modules()) == 2
    assert not any(isinstance(m, nn.ReLU) for m in encoder.modules())
    assert encoder.feature_dim == 10
    out = encoder(torch.full((3, 6), 1e6))
    assert out.shape == (3, 10)
    assert out.abs().max() <= 1.0  # tanh-bounded


def test_mlp_encoder_without_hidden_layers_is_a_flatten():
    encoder = MLPEncoder((2, 3), hidden=[])
    assert encoder.feature_dim == 6
    obs = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    torch.testing.assert_close(encoder(obs), obs.reshape(2, 6))


def test_cnn_and_grid_encoders_use_relu_everywhere():
    for encoder in (NatureCNN((4, 84, 84), 16), GridEncoder((14, 15, 16), 16)):
        weighted = [m for m in encoder.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        relus = [m for m in encoder.modules() if isinstance(m, nn.ReLU)]
        assert len(relus) == len(weighted)
        assert not any(isinstance(m, nn.Tanh) for m in encoder.modules())
        assert encoder(torch.zeros((1, *encoder.obs_shape))).min() >= 0.0


# --------------------------------------------------------------------------- #
# explicit encoder override
# --------------------------------------------------------------------------- #


def test_explicit_mlp_override_flattens_a_grid():
    encoder, dim = build_encoder(GRID_SPACE, NetworkConfig(encoder="mlp", mlp_hidden=[24]))
    assert isinstance(encoder, MLPEncoder)
    assert dim == 24
    first = next(m for m in encoder.modules() if isinstance(m, nn.Linear))
    assert first.in_features == 14 * 15 * 16
    assert encoder(make_batch(GRID_SPACE, 2)).shape == (2, 24)


def test_explicit_grid_override_on_a_small_uint8_image():
    space = gym.spaces.Box(0, 255, (2, 20, 20), np.uint8)
    with pytest.raises(ValueError):  # auto picks the Nature CNN, which needs >= 36 px
        build_encoder(space, NetworkConfig())
    encoder, dim = build_encoder(space, NetworkConfig(encoder="grid", hidden_size=40))
    assert isinstance(encoder, GridEncoder)
    assert dim == 40
    obs = make_batch(space, 3)
    torch.testing.assert_close(encoder(obs), encoder(obs.float() / 255.0))


def test_explicit_cnn_override_on_a_float_image():
    space = gym.spaces.Box(0.0, 1.0, (3, 40, 44), np.float32)
    assert isinstance(build_encoder(space, NetworkConfig())[0], GridEncoder)
    encoder, dim = build_encoder(space, NetworkConfig(encoder="cnn", hidden_size=40))
    assert isinstance(encoder, NatureCNN)
    assert encoder(make_batch(space, 2)).shape == (2, dim)


def test_encoder_name_is_case_insensitive_and_unknown_names_raise():
    assert isinstance(build_encoder(VECTOR_SPACE, NetworkConfig(encoder="MLP"))[0], MLPEncoder)
    with pytest.raises(ValueError, match="resnet"):
        build_encoder(VECTOR_SPACE, NetworkConfig(encoder="resnet"))


# --------------------------------------------------------------------------- #
# bad spaces / shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "space",
    [
        pytest.param(gym.spaces.Box(0, 255, (84, 84), np.uint8), id="2d"),
        pytest.param(gym.spaces.Box(0, 255, (2, 4, 84, 84), np.uint8), id="4d"),
        pytest.param(gym.spaces.Box(0.0, 1.0, (), np.float32), id="scalar"),
        pytest.param(gym.spaces.Box(0, 255, (240, 256, 3), np.uint8), id="channel-last"),
        pytest.param(gym.spaces.Box(0, 9, (4, 84, 84), np.int32), id="int32-image"),
    ],
)
def test_auto_rejects_unsupported_observation_spaces(space):
    with pytest.raises(ValueError):
        build_encoder(space, NetworkConfig())


@pytest.mark.parametrize("name", ["cnn", "grid"])
def test_conv_encoders_reject_non_image_shapes(name):
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        build_encoder(VECTOR_SPACE, NetworkConfig(encoder=name))


def test_non_box_space_raises_type_error():
    with pytest.raises(TypeError, match="Box"):
        build_encoder(gym.spaces.Discrete(4), NetworkConfig())


def test_invalid_sizes_raise():
    with pytest.raises(ValueError, match="hidden_size"):
        build_encoder(GRID_SPACE, NetworkConfig(hidden_size=0))
    with pytest.raises(ValueError, match="mlp_hidden"):
        build_encoder(VECTOR_SPACE, NetworkConfig(mlp_hidden=[64, 0]))
    with pytest.raises(ValueError, match="n_actions"):
        ActorCritic(VECTOR_SPACE, 0, SMALL)
    with pytest.raises(ValueError, match="n_actions"):
        QNetwork(VECTOR_SPACE, 0, SMALL)


@pytest.mark.parametrize("space", ALL_SPACES)
def test_single_observation_without_batch_dim_raises(space):
    # conv2d and linear both accept unbatched input, so without the check this would silently
    # return features of the wrong rank.
    single = make_batch(space, 1)[0]
    encoder, _ = build_encoder(space, SMALL)
    with pytest.raises(ValueError, match="batch"):
        encoder(single)
    with pytest.raises(ValueError, match="batch"):
        ActorCritic(space, N_ACTIONS, SMALL).get_action_and_value(single)
    with pytest.raises(ValueError, match="batch"):
        ActorCritic(space, N_ACTIONS, SMALL).get_value(single)
    with pytest.raises(ValueError, match="batch"):
        QNetwork(space, N_ACTIONS, SMALL)(single)


def test_mismatched_observation_shape_raises():
    encoder, _ = build_encoder(GRID_SPACE, SMALL)
    with pytest.raises(ValueError, match="14, 15, 16"):
        encoder(torch.zeros((2, 14, 16, 15)))
    with pytest.raises(ValueError):
        encoder(torch.zeros((2, 3, 14, 15, 16)))
    vec_encoder, _ = build_encoder(VECTOR_SPACE, SMALL)
    with pytest.raises(ValueError):
        vec_encoder(torch.zeros((2, 5)))


def test_numpy_input_raises_type_error():
    encoder, _ = build_encoder(VECTOR_SPACE, SMALL)
    with pytest.raises(TypeError, match="Tensor"):
        encoder(np.zeros((2, 4), dtype=np.float32))


# --------------------------------------------------------------------------- #
# input dtype handling
# --------------------------------------------------------------------------- #


def test_uint8_images_are_scaled_by_255_inside_the_encoder():
    torch.manual_seed(0)
    encoder, _ = build_encoder(IMAGE_SPACE, SMALL)
    obs = make_batch(IMAGE_SPACE, 2)
    out = encoder(obs)
    torch.testing.assert_close(out, encoder(obs.float() / 255.0))
    assert not torch.allclose(out, encoder(obs.float()))  # floats pass through unscaled
    assert obs.dtype == torch.uint8  # the input is not modified


def test_scaled_input_range_is_zero_to_one():
    # With all-255 input the first conv sees exactly 1.0 everywhere.
    encoder = NatureCNN((1, 36, 36), hidden_size=8)
    seen = []
    first_conv = next(m for m in encoder.modules() if isinstance(m, nn.Conv2d))
    first_conv.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    encoder(torch.full((1, 1, 36, 36), 255, dtype=torch.uint8))
    assert seen[0].dtype == torch.float32
    assert torch.all(seen[0] == 1.0)


@pytest.mark.parametrize("space", [GRID_SPACE, VECTOR_SPACE], ids=["grid", "vector"])
def test_float_inputs_pass_through_and_float64_is_cast(space):
    encoder, _ = build_encoder(space, SMALL)
    obs = make_batch(space, 3)
    seen = []
    first = next(m for m in encoder.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    first.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    out = encoder(obs)
    torch.testing.assert_close(seen[0].reshape(obs.shape), obs, atol=0, rtol=0)
    out64 = encoder(obs.double())
    assert out64.dtype == torch.float32
    torch.testing.assert_close(out64, out)


@pytest.mark.parametrize("space", ALL_SPACES)
def test_all_parameters_and_outputs_are_float32(space):
    ac = ActorCritic(space, N_ACTIONS, SMALL)
    qnet = QNetwork(space, N_ACTIONS, SMALL, dueling=True)
    assert {p.dtype for p in ac.parameters()} == {torch.float32}
    assert {p.dtype for p in qnet.parameters()} == {torch.float32}
    obs = make_batch(space)
    action, log_prob, entropy, value = ac.get_action_and_value(obs)
    assert action.dtype == torch.int64
    assert log_prob.dtype == entropy.dtype == value.dtype == torch.float32
    assert qnet(obs).dtype == torch.float32


# --------------------------------------------------------------------------- #
# initialisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("space", ALL_SPACES)
def test_actor_critic_orthogonal_init(space):
    torch.manual_seed(0)
    model = ActorCritic(space, N_ACTIONS, SMALL)
    eye = torch.eye(N_ACTIONS)
    torch.testing.assert_close(
        rows_gram(model.policy_head.weight), 0.01**2 * eye, atol=1e-7, rtol=0
    )
    torch.testing.assert_close(rows_gram(model.value_head.weight), torch.eye(1), atol=1e-5, rtol=0)
    heads = {model.policy_head, model.value_head}
    hidden = [
        m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)) and m not in heads
    ]
    assert hidden
    for layer in hidden:
        gram = rows_gram(layer.weight)
        rows, cols = layer.weight.flatten(1).shape
        if rows <= cols:  # rows can only be orthonormal when there are no more rows than columns
            torch.testing.assert_close(gram, 2.0 * torch.eye(rows), atol=1e-4, rtol=0)
        else:
            cols_gram = layer.weight.detach().flatten(1).T @ layer.weight.detach().flatten(1)
            torch.testing.assert_close(cols_gram, 2.0 * torch.eye(cols), atol=1e-4, rtol=0)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            assert torch.all(module.bias == 0.0)


@pytest.mark.parametrize("space", ALL_SPACES)
def test_initial_policy_is_near_uniform(space):
    torch.manual_seed(1)
    model = ActorCritic(space, N_ACTIONS, NetworkConfig())
    logits, _ = model(make_batch(space, 8))
    probs = torch.softmax(logits, dim=-1)
    assert (probs - 1.0 / N_ACTIONS).abs().max() < 0.02
    _, _, entropy, _ = model.get_action_and_value(make_batch(space, 8))
    torch.testing.assert_close(entropy, torch.full((8,), math.log(N_ACTIONS)), atol=1e-3, rtol=0)


def test_q_network_heads_are_orthogonal_with_unit_gain():
    torch.manual_seed(0)
    plain = QNetwork(VECTOR_SPACE, N_ACTIONS, SMALL)
    torch.testing.assert_close(
        rows_gram(plain.q_head.weight), torch.eye(N_ACTIONS), atol=1e-5, rtol=0
    )
    dueling = QNetwork(VECTOR_SPACE, N_ACTIONS, SMALL, dueling=True)
    torch.testing.assert_close(
        rows_gram(dueling.advantage_head.weight), torch.eye(N_ACTIONS), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        rows_gram(dueling.value_head.weight), torch.eye(1), atol=1e-5, rtol=0
    )


def test_construction_is_deterministic_given_the_torch_seed():
    torch.manual_seed(123)
    a = ActorCritic(GRID_SPACE, N_ACTIONS, SMALL)
    torch.manual_seed(123)
    b = ActorCritic(GRID_SPACE, N_ACTIONS, SMALL)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


# --------------------------------------------------------------------------- #
# ActorCritic
# --------------------------------------------------------------------------- #


def sharpen_policy(model: ActorCritic) -> None:
    """Re-initialise the policy head with a large gain so that the policy is far from uniform."""
    torch.manual_seed(7)
    layer_init(model.policy_head, std=3.0)


@pytest.mark.parametrize("space", ALL_SPACES)
def test_get_action_and_value_is_consistent_with_categorical(space):
    torch.manual_seed(0)
    model = ActorCritic(space, N_ACTIONS, SMALL)
    sharpen_policy(model)
    obs = make_batch(space, 6)
    action, log_prob, entropy, value = model.get_action_and_value(obs)
    assert action.shape == log_prob.shape == entropy.shape == value.shape == (6,)
    assert action.min() >= 0 and action.max() < N_ACTIONS

    logits, forward_value = model(obs)
    assert logits.shape == (6, N_ACTIONS)
    dist = Categorical(logits=logits)
    torch.testing.assert_close(log_prob, dist.log_prob(action))
    torch.testing.assert_close(entropy, dist.entropy())
    torch.testing.assert_close(value, forward_value)
    torch.testing.assert_close(value, model.get_value(obs))
    assert model.get_value(obs).shape == (6,)


def test_given_action_is_returned_unchanged_and_evaluated():
    torch.manual_seed(0)
    model = ActorCritic(GRID_SPACE, N_ACTIONS, SMALL)
    sharpen_policy(model)
    obs = make_batch(GRID_SPACE, 6)
    given = torch.tensor([0, 6, 3, 3, 1, 5])
    action, log_prob, entropy, _ = model.get_action_and_value(obs, action=given)
    assert action is given
    logits, _ = model(obs)
    expected = torch.log_softmax(logits, dim=-1).gather(1, given.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(log_prob, expected)
    torch.testing.assert_close(entropy, Categorical(logits=logits).entropy())
    # A given action wins over `deterministic`.
    action_det, log_prob_det, _, _ = model.get_action_and_value(obs, given, deterministic=True)
    assert action_det is given
    torch.testing.assert_close(log_prob_det, expected)


@pytest.mark.parametrize("bad", [torch.zeros((6, 1), dtype=torch.long), torch.zeros(5).long()])
def test_given_action_with_wrong_shape_raises(bad):
    # A (B, 1) action would silently broadcast to (B, B) log-probs inside Categorical.
    model = ActorCritic(VECTOR_SPACE, N_ACTIONS, SMALL)
    with pytest.raises(ValueError, match="action"):
        model.get_action_and_value(make_batch(VECTOR_SPACE, 6), action=bad)


def test_deterministic_action_is_the_argmax():
    torch.manual_seed(0)
    model = ActorCritic(VECTOR_SPACE, N_ACTIONS, SMALL)
    sharpen_policy(model)
    obs = make_batch(VECTOR_SPACE, 32)
    logits, _ = model(obs)
    assert logits.argmax(dim=-1).unique().numel() > 1  # the check below is not trivial
    for _ in range(3):
        action, log_prob, _, _ = model.get_action_and_value(obs, deterministic=True)
        assert torch.equal(action, logits.argmax(dim=-1))
        torch.testing.assert_close(log_prob, torch.log_softmax(logits, -1).max(dim=-1).values)


def test_sampled_actions_follow_the_policy_distribution():
    torch.manual_seed(0)
    model = ActorCritic(VECTOR_SPACE, 3, SMALL)
    with torch.no_grad():
        model.policy_head.bias.copy_(torch.log(torch.tensor([0.7, 0.2, 0.1])))
    obs = torch.zeros((6000, 4))
    action, _, _, _ = model.get_action_and_value(obs)
    freq = torch.bincount(action, minlength=3).float() / 6000
    torch.testing.assert_close(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.03, rtol=0)
    # Sampling consumes the global torch RNG, so it is reproducible under a seed.
    torch.manual_seed(5)
    first = model.get_action_and_value(obs[:64])[0]
    torch.manual_seed(5)
    assert torch.equal(first, model.get_action_and_value(obs[:64])[0])


def test_mlp_actor_and_critic_do_not_share_a_trunk_but_conv_encoders_are_shared():
    assert ActorCritic(IMAGE_SPACE, N_ACTIONS, SMALL).critic_encoder is None
    assert ActorCritic(GRID_SPACE, N_ACTIONS, SMALL).critic_encoder is None
    model = ActorCritic(VECTOR_SPACE, N_ACTIONS, SMALL)
    assert isinstance(model.critic_encoder, MLPEncoder)
    assert model.critic_encoder is not model.encoder
    obs = make_batch(VECTOR_SPACE, 4)
    logits_before, value_before = model(obs)
    with torch.no_grad():
        for param in model.encoder.parameters():
            param.add_(0.5)
    logits_after, value_after = model(obs)
    assert not torch.allclose(logits_before, logits_after)
    torch.testing.assert_close(value_before, value_after, atol=0, rtol=0)


@pytest.mark.parametrize("space", ALL_SPACES)
def test_gradients_flow_to_all_actor_critic_parameters(space):
    torch.manual_seed(0)
    model = ActorCritic(space, N_ACTIONS, SMALL)
    sharpen_policy(model)
    _, log_prob, entropy, value = model.get_action_and_value(make_batch(space, 8))
    (log_prob.sum() + entropy.sum() + value.sum()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_get_value_only_needs_the_critic_path():
    torch.manual_seed(0)
    model = ActorCritic(VECTOR_SPACE, N_ACTIONS, SMALL)
    model.get_value(make_batch(VECTOR_SPACE, 4)).sum().backward()
    assert model.policy_head.weight.grad is None
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None for p in model.critic_encoder.parameters())


@pytest.mark.parametrize("space", ALL_SPACES)
def test_actor_critic_state_dict_round_trip(space):
    torch.manual_seed(0)
    source = ActorCritic(space, N_ACTIONS, SMALL)
    torch.manual_seed(1)
    target = ActorCritic(space, N_ACTIONS, SMALL)
    obs = make_batch(space, 4)
    assert not torch.equal(source(obs)[1], target(obs)[1])
    target.load_state_dict(source.state_dict())
    for out_source, out_target in zip(source(obs), target(obs), strict=True):
        assert torch.equal(out_source, out_target)


def test_actor_critic_on_a_real_gymnasium_env():
    env = gym.make("CartPole-v1")
    try:
        obs, _ = env.reset(seed=0)
        model = ActorCritic(env.observation_space, int(env.action_space.n), NetworkConfig())
        batch = torch.as_tensor(obs).unsqueeze(0)
        action, log_prob, _, value = model.get_action_and_value(batch)
        assert env.action_space.contains(int(action.item()))
        assert log_prob.shape == value.shape == (1,)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# QNetwork
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dueling", [False, True])
@pytest.mark.parametrize("space", ALL_SPACES)
def test_q_network_output_shape(space, dueling):
    qnet = QNetwork(space, N_ACTIONS, SMALL, dueling=dueling)
    assert qnet.dueling is dueling
    assert qnet(make_batch(space, 5)).shape == (5, N_ACTIONS)


def test_plain_q_network_is_a_linear_head_on_the_encoder():
    torch.manual_seed(0)
    qnet = QNetwork(GRID_SPACE, N_ACTIONS, SMALL)
    assert not hasattr(qnet, "advantage_head")
    obs = make_batch(GRID_SPACE, 3)
    torch.testing.assert_close(qnet(obs), qnet.q_head(qnet.encoder(obs)))


@pytest.mark.parametrize("space", ALL_SPACES)
def test_dueling_identity(space):
    torch.manual_seed(0)
    qnet = QNetwork(space, N_ACTIONS, SMALL, dueling=True)
    assert not hasattr(qnet, "q_head")
    obs = make_batch(space, 5)
    features = qnet.encoder(obs)
    value = qnet.value_head(features)
    advantage = qnet.advantage_head(features)
    assert value.shape == (5, 1) and advantage.shape == (5, N_ACTIONS)
    q = qnet(obs)
    torch.testing.assert_close(q, value + advantage - advantage.mean(dim=1, keepdim=True))
    # Consequences of the identity: mean_a Q = V, and Q - mean_a Q = A - mean_a A.
    torch.testing.assert_close(q.mean(dim=1, keepdim=True), value, atol=1e-5, rtol=0)
    assert q.std(dim=1).min() > 0  # the advantage stream is not cancelled out


@pytest.mark.parametrize("dueling", [False, True])
@pytest.mark.parametrize("space", ALL_SPACES)
def test_gradients_flow_to_all_q_network_parameters(space, dueling):
    torch.manual_seed(0)
    qnet = QNetwork(space, N_ACTIONS, SMALL, dueling=dueling)
    obs = make_batch(space, 8)
    actions = torch.arange(8) % N_ACTIONS
    chosen = qnet(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    (chosen - 1.0).pow(2).mean().backward()
    for name, param in qnet.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_q_network_state_dict_copies_into_a_target_network():
    torch.manual_seed(0)
    online = QNetwork(IMAGE_SPACE, N_ACTIONS, SMALL, dueling=True)
    torch.manual_seed(1)
    target = QNetwork(IMAGE_SPACE, N_ACTIONS, SMALL, dueling=True)
    target.load_state_dict(online.state_dict())
    obs = make_batch(IMAGE_SPACE, 2)
    assert torch.equal(online(obs), target(obs))


# --------------------------------------------------------------------------- #
# accelerator smoke test (skipped where MPS is unavailable)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
@pytest.mark.parametrize("space", ALL_SPACES)
def test_networks_run_on_mps_in_float32(space):
    device = torch.device("mps")
    torch.manual_seed(0)
    model = ActorCritic(space, N_ACTIONS, SMALL)
    qnet = QNetwork(space, N_ACTIONS, SMALL, dueling=True)
    obs = make_batch(space, 4)
    cpu_logits, cpu_value = model(obs)
    cpu_q = qnet(obs)
    model.to(device)
    qnet.to(device)
    action, log_prob, entropy, value = model.get_action_and_value(obs.to(device))
    assert action.device.type == "mps" and action.shape == (4,)
    assert log_prob.dtype == entropy.dtype == value.dtype == torch.float32
    logits, _ = model(obs.to(device))
    torch.testing.assert_close(logits.cpu(), cpu_logits, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(value.cpu(), cpu_value, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(qnet(obs.to(device)).cpu(), cpu_q, atol=1e-4, rtol=1e-4)
