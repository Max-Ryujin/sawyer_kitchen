from typing import Iterable, Sequence, Tuple
import numpy as np
import mujoco as mj
from typing import Optional, Sequence, NamedTuple

def _try_name_lookup(env, name: str, obj_type: str):
    """Return id or -1 if not found."""
    lookup_map = {
        'body': mj.mjtObj.mjOBJ_BODY,
        'site': mj.mjtObj.mjOBJ_SITE,
        'geom': mj.mjtObj.mjOBJ_GEOM,
        'joint': mj.mjtObj.mjOBJ_JOINT,
        'actuator': mj.mjtObj.mjOBJ_ACTUATOR,
    }
    if obj_type not in lookup_map:
        return -1
    return mj.mj_name2id(env.unwrapped.model, lookup_map[obj_type], name)


def get_effector_pos(env, site_candidates=('endeffector0', 'right_wrist')) -> np.ndarray:
    """Return (x, y, z) world position of likely end-effector site or body."""
    data, model = env.unwrapped.data, env.unwrapped.model
    for name in site_candidates:
        sid = _try_name_lookup(env, name, 'site')
        if sid != -1:
            return np.asarray(data.site_xpos[sid]).copy()
    for name in site_candidates:
        bid = _try_name_lookup(env, name, 'body')
        if bid != -1:
            return np.asarray(data.xpos[bid]).copy()
    raise ValueError(f"No matching EE site/body found: {site_candidates}")


def get_object_pos(env, name_candidates=('cup_freejoint1', 'cup1')) -> np.ndarray:
    """Return (x, y, z) of the first matching object."""
    model, data = env.unwrapped.model, env.unwrapped.data
    for name in name_candidates:
        jid = _try_name_lookup(env, name, 'joint')
        if jid != -1:
            addr = int(model.jnt_qposadr[jid])
            return np.asarray(data.qpos[addr:addr+3]).copy()
        bid = _try_name_lookup(env, name, 'body')
        if bid != -1:
            return np.asarray(data.xpos[bid]).copy()
    raise ValueError(f"No object found with names {name_candidates}")


def make_gripper_action(env, close=True, close_val=1.0, open_val=-1.0) -> np.ndarray:
    """Return array setting gripper actuator commands."""
    nu = int(env.nu)
    a = np.zeros(nu, dtype=np.float32)
    val = float(close_val if close else open_val)

    set_indices = []
    for pname in ("rc_close", "lc_close"):
        aid = _try_name_lookup(env, pname, 'actuator')
        if aid != -1:
            set_indices.append(int(aid))

    # Fallback indices
    if not set_indices:
        set_indices = [7, 8] if nu >= 9 else []

    for i in set_indices:
        a[i] = val
    return a


def make_joint_pd_action(env, target_qpos, kp=8.0, joint_slice=slice(0, 7)) -> np.ndarray:
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
    """Inverse kinematics using Jacobian pseudoinverse."""
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    nv = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    
    # err size depends on whether rotation is used
    err = np.zeros(6 if target_quat is not None else 3)

    for step in range(max_steps):
        mj.mj_fwdPosition(model, data)
        site_xpos = data.site_xpos[site_id].copy()
        site_xmat = data.site_xmat[site_id].reshape(3, 3)

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
            return IKResult(data.qpos.copy(), np.linalg.norm(err), step, True)

        mj.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr]) if target_quat is not None else jacp

        # Fix: slice err to match Jacobian
        dq = nullspace_method(jac, err[:jac.shape[0]], reg_strength)
        data.qpos[:nv] += dq  # only first nv elements

    return IKResult(data.qpos.copy(), np.linalg.norm(err), step, False)
