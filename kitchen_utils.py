from typing import Iterable, Sequence, Tuple
import numpy as np
import mujoco as mj
from typing import Optional, Sequence, NamedTuple


def _try_name_lookup(env, name: str, obj_type: str):
    """Return id"""
    lookup_map = {
        "body": mj.mjtObj.mjOBJ_BODY,
        "site": mj.mjtObj.mjOBJ_SITE,
        "geom": mj.mjtObj.mjOBJ_GEOM,
        "joint": mj.mjtObj.mjOBJ_JOINT,
        "actuator": mj.mjtObj.mjOBJ_ACTUATOR,
    }
    if obj_type not in lookup_map:
        raise ValueError(f"Unknown object type: {obj_type}")
    return mj.mj_name2id(env.unwrapped.model, lookup_map[obj_type], name)


def get_effector_pos(env, site_candidates=("grip_site",)) -> np.ndarray:
    """Return (x, y, z) world position of likely end-effector site or body."""
    data, model = env.unwrapped.data, env.unwrapped.model
    for name in site_candidates:
        sid = _try_name_lookup(env, name, "site")
        if sid != -1:
            return np.asarray(data.site_xpos[sid]).copy()
    for name in site_candidates:
        bid = _try_name_lookup(env, name, "body")
        if bid != -1:
            return np.asarray(data.xpos[bid]).copy()
    raise ValueError(f"No matching EE site/body found: {site_candidates}")


def get_object_pos(env, name_candidates=("cup_freejoint1", "cup1")) -> np.ndarray:
    """Return (x, y, z) of the first matching object."""
    model, data = env.unwrapped.model, env.unwrapped.data
    for name in name_candidates:
        jid = _try_name_lookup(env, name, "joint")
        if jid != -1:
            addr = int(model.jnt_qposadr[jid])
            return np.asarray(data.qpos[addr : addr + 3]).copy()
        bid = _try_name_lookup(env, name, "body")
        if bid != -1:
            return np.asarray(data.xpos[bid]).copy()
    raise ValueError(f"No object found with names {name_candidates}")


def make_gripper_action(env, close=True, close_val=0.0, open_val=0.015) -> np.ndarray:
    """Return array setting gripper actuator commands."""
    nu = int(env.unwrapped.nu)
    a = np.zeros(nu, dtype=np.float32)
    val = float(close_val if close else open_val)

    set_indices = []
    for pname in ("rc_close", "lc_close"):
        aid = _try_name_lookup(env, pname, "joint")
        if aid != -1:
            set_indices.append(int(aid))

    for i in set_indices:
        a[i] = val
    return a


def make_joint_pd_action(
    env, target_qpos, kp=8.0, joint_slice=slice(0, 7)
) -> np.ndarray:
    """Simple proportional control toward target joint qpos."""
    nu = int(env.unwrapped.nu)
    a = np.zeros(nu, dtype=np.float32)
    qpos = np.asarray(env.unwrapped.data.qpos)
    tgt = np.asarray(target_qpos, dtype=float)
    cur = qpos[joint_slice]
    n = min(len(tgt), cur.shape[0])
    cmd = np.clip(kp * (tgt[:n] - cur[:n]), -1, 1)
    start = joint_slice.start or 0
    for i in range(n):
        idx = start + i
        if 0 <= idx < nu:
            a[idx] = float(cmd[i])
    return a


class IKResult(NamedTuple):
    qpos: np.ndarray
    err_norm: float
    steps: int
    success: bool


def nullspace_method(jac_joints: np.ndarray, delta: np.ndarray, reg: float = 0.0):
    """Damped least squares for Jacobian-based IK."""
    hess = jac_joints.T @ jac_joints
    rhs = jac_joints.T @ delta
    if reg > 0:
        hess += np.eye(hess.shape[0]) * reg
        return np.linalg.solve(hess, rhs)
    return np.linalg.lstsq(hess, rhs, rcond=None)[0]


def qpos_from_site_pose(
    model: mj.MjModel,
    data: mj.MjData,
    site_name: str,
    target_pos: np.ndarray,
    target_quat: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    rot_weight: float = 1.0,
    reg_strength: float = 3e-2,
    max_steps: int = 100,
) -> IKResult:
    # Create a copy of the data to avoid modifying the simulation state
    data_copy = mj.MjData(model)
    """Inverse kinematics using Jacobian pseudoinverse."""
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site {site_name} not found in model")

    data_copy.qpos[:] = data.qpos
    data_copy.qvel[:] = data.qvel
    data_copy.act[:] = data.act
    data_copy.ctrl[:] = data.ctrl

    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))  # err size depends on whether rotation is used
    err = np.zeros(6 if target_quat is not None else 3)

    for step in range(max_steps):
        mj.mj_fwdPosition(model, data_copy)
        site_xpos = data_copy.site_xpos[site_id].copy()
        site_xmat = data_copy.site_xmat[site_id].reshape(3, 3)

        err[:3] = target_pos - site_xpos

        if target_quat is not None:
            site_quat = np.empty(4)
            mj.mju_mat2Quat(site_quat, site_xmat)
            diff_quat = np.empty(4)
            mj.mju_negQuat(diff_quat, site_quat)
            mj.mju_mulQuat(diff_quat, target_quat, diff_quat)
            mj.mju_quat2Vel(err[3:], diff_quat, 1)
            err[3:] *= rot_weight

        # check convergence
        if np.linalg.norm(err) < tol:
            return IKResult(data_copy.qpos.copy(), np.linalg.norm(err), step, True)

        mj.mj_jacSite(model, data_copy, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr]) if target_quat is not None else jacp

        # Fix: slice err to match Jacobian
        dq = nullspace_method(jac, err[: jac.shape[0]], reg_strength)
        data_copy.qpos[:nv] += dq  # only first nv elements

    return IKResult(data_copy.qpos.copy(), np.linalg.norm(err), step, False)


def ik_step(
    model: mj.MjModel,
    data: mj.MjData,
    site_name: str,
    target_pos: np.ndarray,
    target_quat: Optional[np.ndarray] = None,
    rot_weight: float = 1.0,
    reg_strength: float = 3e-3,
) -> np.ndarray:
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site {site_name} not found in model")

    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    err = np.zeros(6 if target_quat is not None else 3)

    # Forward kinematics
    # mj.mj_fwdPosition(model, data)

    # Compute current site position/orientation
    site_xpos = data.site_xpos[site_id].copy()
    site_xmat = data.site_xmat[site_id].reshape(3, 3)

    # Compute position error
    err[:3] = target_pos - site_xpos
    err_norm = np.linalg.norm(err[:3])

    # Optional orientation error
    if target_quat is not None:
        site_quat = np.empty(4)
        mj.mju_mat2Quat(site_quat, site_xmat.reshape(9))
        diff_quat = np.empty(4)
        mj.mju_negQuat(diff_quat, site_quat)
        mj.mju_mulQuat(diff_quat, target_quat, diff_quat)
        mj.mju_quat2Vel(err[3:], diff_quat, 1)
        err[3:] *= rot_weight

    # Compute Jacobian
    mj.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr]) if target_quat is not None else jacp

    dq = nullspace_method(jac, err[: jac.shape[0]], reg_strength)
    return dq[:7]


import numpy as np
import mujoco as mj
import copy


def ik_solve_dm(
    model: mj.MjModel,
    data: mj.MjData,
    site_name: str,
    target_pos: np.ndarray = None,
    target_quat: np.ndarray = None,
    joint_indices: np.ndarray = None,
    tol: float = 1e-5,
    rot_weight: float = 1.0,
    regularization_threshold: float = 1e-2,
    regularization_strength: float = 3e-2,
    max_update_norm: float = 1.0,
    progress_thresh: float = 20.0,
    max_steps: int = 100,
    inplace: bool = True,
) -> tuple[np.ndarray, float, int, bool]:
    """Iteratively solve IK for a target site pose. Returns (qpos, err_norm, steps, success)."""
    assert (
        target_pos is not None or target_quat is not None
    ), "Must provide at least target_pos or target_quat."

    if not inplace:
        #data_copy = copy.copy(data)
        #data = data_copy
        data_copy = mj.MjData(model)
        data_copy.qpos[:] = data.qpos
        data_copy.qvel[:] = data.qvel
        data_copy.act[:]  = data.act
        data = data_copy

    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site {site_name} not found in model")

    nv = model.nv
    qpos = data.qpos.copy()
    update = np.zeros(nv)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    err = np.zeros(6 if target_quat is not None else 3)
    site_quat = np.empty(4)
    neg_quat = np.empty(4)
    err_quat = np.empty(4)

    if joint_indices is None:
        joint_indices = slice(None)

    success = False
    err_norm = np.inf

    for step in range(max_steps):
        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        site_xpos = data.site_xpos[site_id].copy()
        site_xmat = data.site_xmat[site_id].reshape(3, 3)

        # --- compute error ---
        err[:3] = target_pos - site_xpos if target_pos is not None else 0.0
        err_norm = np.linalg.norm(err[:3])

        if target_quat is not None:
            mj.mju_mat2Quat(site_quat, site_xmat.reshape(9))
            mj.mju_negQuat(neg_quat, site_quat)
            mj.mju_mulQuat(err_quat, target_quat, neg_quat)
            mj.mju_quat2Vel(err[3:], err_quat, 1)
            err[3:] *= rot_weight
            err_norm += np.linalg.norm(err[3:]) * rot_weight

        if err_norm < tol:
            success = True
            break

        # --- compute Jacobian ---
        mj.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr]) if target_quat is not None else jacp
        jac = jac[:, joint_indices]

        # --- damping / regularization ---
        reg = regularization_strength if err_norm > regularization_threshold else 0.0
        hess = jac.T @ jac
        rhs = jac.T @ err[: jac.shape[0]]
        if reg > 0:
            hess += np.eye(hess.shape[0]) * reg
            dq = np.linalg.solve(hess, rhs)
        else:
            dq = np.linalg.lstsq(hess, rhs, rcond=None)[0]

        update_norm = np.linalg.norm(dq)
        if update_norm > max_update_norm:
            dq *= max_update_norm / update_norm

        if update_norm > 1e-12:
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                break

        update[:] = 0.0
        update[joint_indices] = dq
        mj.mj_integratePos(model, qpos, update, 1)

    return qpos


def get_body_contact_force(
    model: mj.MjModel, data: mj.MjData, body_name: str
) -> np.ndarray:
    """
    Calculates the total contact force vector acting on a specified body.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        body_name: The name of the body to check for contacts.

    Returns:
        A 3D numpy array representing the total contact force vector in the world frame.
    """
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return np.zeros(3)

    total_force = np.zeros(3)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == body_id or contact.geom2 == body_id:
            force_vector = np.zeros(6)
            mj.mj_contactForce(model, data, i, force_vector)
            total_force += force_vector[:3]  # We only care about the linear force
    return total_force
