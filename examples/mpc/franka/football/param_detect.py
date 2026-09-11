import signal
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

import time
import json
import numpy as np
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from argparse import Namespace
import warnings
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.transform import Rotation

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

# 导入原有模块
from examples.mpc.fingertips.test.params import ExplicitMPCParams
from planning.mpc_explicit2 import MPCExplicit
from planning.mpc_implicit import MPCImplicit
from envs.fingertips_env import MjSimulator
from contact.fingertips_collision_detection2 import Contact
from utils import metrics, rotations

class HyperparameterOptimizer:
    def __init__(self, base_args: Namespace, num_trials: int = 100, max_workers: int = 4, max_param_num: int = 20):
        """
        初始化超参数优化器（多进程安全版本）
        
        参数:
            base_args: 基础参数配置
            num_trials: 每组参数的试验次数
            max_workers: 最大并行工作进程数
            max_param_num: 最大参数组合数
        """
        self.base_args = base_args
        self.num_trials = num_trials
        self.max_workers = max_workers
        self.max_param_num = max_param_num
        self.results = []
        self.best_params = None
        self.best_success_rate = 0
        self.start_time = None
        self.total_combinations = 0
        self.max_rate_threshold = 0.55
        self.min_rate_threshold = 0.4
        self.delta_threshold = 0.01
        self.tested_param_num = 0

        # 创建结果目录
        self.save_dir = './param_search_results/' + base_args.obj
        os.makedirs(self.save_dir, exist_ok=True)
    
    def generate_parameters(self) -> List[Namespace]:
        """生成要测试的参数组合"""
        param_combinations = []
        
        # 定义参数搜索空间
        attract_coefs = [0.5]
        reject_coefs = [0.001, 0.0005]  # important
        contact_coefs = [0.5, 0.3, 0.7]  # important
        contact_cost_param = [1]
        model_params = [7, 5, 6, 10]  # important
        reject_dis_values = [0.02, 0.01]
        attract_point_comps = [0.1]
        ori_coef = [0.001, 0, 0.005]
        low_err_coer = [0.3, 0.1]
        
        # 生成所有组合
        for ac in attract_coefs:
            for rc in reject_coefs:
                for cc in contact_coefs:
                    for mp in model_params:
                        for rd in reject_dis_values:
                            for apc in attract_point_comps:
                                for orc in ori_coef:
                                    for ccp in contact_cost_param:
                                        for lec in low_err_coer:
                                            args = Namespace(**vars(self.base_args))
                                            args.attract_coef = ac
                                            args.reject_coef = rc
                                            args.contact_coef = cc
                                            args.model_param = mp
                                            args.reject_dis = rd
                                            args.attract_point_comp = apc
                                            args.ori_coef = orc
                                            args.contact_cost_param = ccp
                                            args.low_err_coef = lec
                                            param_combinations.append(args)
        
        # 如果组合太多，随机抽样
        if len(param_combinations) > self.max_param_num:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:self.max_param_num]
        
        self.total_combinations = len(param_combinations)
        return param_combinations
    
    def _evaluate_single_trial(self, params: Namespace, trial_count: int) -> Dict:
        """执行单个trial的评估（进程安全）"""
        trial_start = time.time()
        trial_result = {
            "trial": trial_count,
            "success": False,
            "steps": 0,
            "time": 0
        }
        
        try:
            # 初始化环境和控制器
            param = ExplicitMPCParams(params, rand_seed=trial_count, target_type='ground-rotation', model='explicit')
            mpc = MPCExplicit(param) if param.mpc_model == 'explicit' else MPCImplicit(param)
            contact = Contact(param)
            env = MjSimulator(param)
            
            # 运行试验
            rollout_step = 0
            consecutive_success_time = 0
            verify_cost = 0
            max_rollout_length = 3000
            consecutive_success_time_threshold = 20
            low_err_coef = params.low_err_coef
            upper_err_coef = params.upper_err_coef
            current_x = np.zeros(7)
            current_x[3] = 1

            while rollout_step < max_rollout_length:
                curr_q = env.get_state()
                
                # 接触检测
                phi_vec, jac_mat, con_point, jac_mat_env = contact.detect_once(env)
                quanternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
                R_obj_to_world = Rotation.from_quat(quanternion).as_matrix()
                gravity = np.hstack([R_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])
                
                delta_quat = rotations.quaternion_multiply(param.target_q_, rotations.quaternion_conjugate(curr_q[3:7]))
                target_pos_ = param.target_p_ - curr_q[0:3]
                target_pos_[2] = 0
                
                target_quat_local = rotations.quaternion_multiply(
                                            rotations.quaternion_conjugate(curr_q[3:7]),
                                            param.target_q_
                                        )
                target_pose_local = np.hstack([R_obj_to_world.T @ target_pos_, target_quat_local])
                
                param.lambda_optimizer.update_Jacobian(jac_mat_env)
                visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                    curr_q[0:3], R_obj_to_world, param.target_p_, params.ground_height_threshold)
                
                best_contact_point, normal, min_error, max_error = param.lambda_optimizer.choose_contact_points(
                    target_pose_local, current_x, gravity, visible_point_idx)
                
                attract_point = best_contact_point.copy()
                attract_point_world = R_obj_to_world @ attract_point + curr_q[0:3]
                original_height = attract_point_world[2]
                attract_point_world -= params.attract_point_comp * R_obj_to_world @ normal
                attract_point_world[2] = max(attract_point_world[2], original_height)
                
                local_point = R_obj_to_world.T @ (curr_q[7:10] - curr_q[0:3])
                p_arm_local, normal_local, x_plus_opt, error, info = param.lambda_optimizer.optimize_control_input(
                    target_pose_local, current_x, gravity, local_point)
                p_arm_world = R_obj_to_world @ p_arm_local + curr_q[:3]

                if verify_cost:
                    low_err_coef = params.low_err_coef
                else:
                    if np.linalg.norm(curr_q[7:10] - attract_point_world) < 5e-2:
                        low_err_coef *= 1.1
                upper_err_coef = max(params.upper_err_coef if not verify_cost else upper_err_coef-0.002, 0.7)

                delta_error = max_error - min_error
                adaptive = delta_error * upper_err_coef if verify_cost else delta_error * low_err_coef
                verify_cost = 1 if error < (min_error + adaptive) else 0
                
                # 规划
                sol = mpc.plan_once(
                    param.target_p_,
                    param.target_q_,
                    curr_q,
                    phi_vec,
                    jac_mat,
                    verify_cost_param=verify_cost,
                    virtual_point=attract_point_world,
                    contact_point=p_arm_world,
                    sol_guess=param.sol_guess_)
                
                param.sol_guess_ = sol['sol_guess']
                action = sol['action']
                
                # 模拟
                env.step(action)
                rollout_step += 1
                
                # 成功检查
                curr_q = env.get_state()
                if (metrics.comp_pos_error(curr_q[0:3], param.target_p_) < 0.02) and \
                   (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < 0.015):
                    consecutive_success_time += 1
                else:
                    consecutive_success_time = 0
                
                # 早期终止
                if consecutive_success_time > consecutive_success_time_threshold:
                    break
            
            # 记录trial结果
            trial_result["success"] = rollout_step < max_rollout_length
            trial_result["steps"] = rollout_step
            
        except Exception as e:
            print(f"\n参数 {self._format_params(params)} 第 {trial_count+1} 次试验失败: {str(e)}")
        
        trial_result["time"] = time.time() - trial_start
        return trial_result
    
    def evaluate_parameters(self, params: Namespace) -> Tuple[Dict, float]:
        """评估一组参数的性能（进程安全版本）"""
        # 获取当前参数组合的索引和剩余数量
        current_index = self.tested_param_num
        remaining = self.total_combinations - current_index
        
        print(f"\n{'='*60}\n"
            f"开始测试参数组合 {current_index}/{self.total_combinations}\n"
            f"剩余组合数: {remaining}\n"
            f"参数: {self._format_params(params)}\n"
            f"{'='*60}")
        
        self.tested_param_num+=1
        trial_results = []
        success_count = 0
        rate_threshold = self.min_rate_threshold
        consecutive_fail_num = 0
        
        for trial_count in range(self.num_trials):
            trial_result = self._evaluate_single_trial(params, trial_count)
            trial_results.append(trial_result)
            
            if trial_result["success"]:
                success_count += 1
                consecutive_fail_num = 0
            else:
                consecutive_fail_num += 1
            
            current_rate = success_count / (trial_count + 1)
            
            # 打印当前trial结果
            print(f"\nTrial {trial_count+1}/{self.num_trials} | "
                f"Params: {self._format_params(params)} | "
                f"结果: {'成功' if trial_result['success'] else '失败'} | "
                f"当前成功率: {current_rate:.1%} | "
                f"耗时: {trial_result['time']} | "
                f"剩余组合数: {remaining}")
            
            # 早期终止检查
            if trial_count >= 10 and current_rate < rate_threshold:
                print(f"早期终止: 参数 {self._format_params(params)} 前{trial_count+1}次成功率低于{rate_threshold*100}%")
                break

            if consecutive_fail_num >= 5:
                print(f"consecutive failure number exceed 5")
                break

            if trial_count % 10 == 0 and trial_count != 0:
                rate_threshold = min(rate_threshold + self.delta_threshold, self.max_rate_threshold)
        
        # 返回完整结果
        success_rate = success_count / (trial_count + 1)
        return {
            "params": vars(params),
            "success_rate": success_rate,
            "trials": trial_results
        }, success_rate

    def _format_params(self, params: Namespace) -> str:
        """简化参数显示"""
        return (f"attract_coef={params.attract_coef}, reject_coef={params.reject_coef}, "
                f"contact_coef={params.contact_coef}, model_param={params.model_param}, "
                f"reject_dis={params.reject_dis}, attract_point_comp={params.attract_point_comp}, "
                f"ori_coef={params.ori_coef}, contact_cost_param={params.contact_cost_param}")

    def run_optimization(self):
        """运行优化（多进程版本）"""
        self.start_time = time.time()
        params_list = self.generate_parameters()
        
        print(f"\n{'='*60}\n"
              f"开始优化 | 总参数组合: {len(params_list)} | 每组试验次数: {self.num_trials}\n"
              f"{'='*60}")
        
        try:
            # 使用进程池
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.evaluate_parameters, p): p for p in params_list}
                
                for future in as_completed(futures):
                    try:
                        result, success_rate = future.result()
                        self.results.append(result)
                        
                        # 更新最佳参数
                        if success_rate > self.best_success_rate:
                            self.best_success_rate = success_rate
                            self.best_params = result["params"]
                        
                        if len(self.results) % 5:
                            self.save_results()
                            
                    except Exception as e:
                        print(f"\n参数测试失败: {str(e)}")
        
        except KeyboardInterrupt:
            print("\n捕获到中断信号，保存当前结果...")
            self.save_results()
            return
        
        # 最终结果
        self.save_results()
        print(f"\n{'='*60}\n优化完成 | 总耗时: {time.time() - self.start_time:.1f}s\n{'='*60}")
        print(f"最佳成功率: {self.best_success_rate:.1%}")
        print(f"最佳参数: {self.best_params}")

    def save_results(self):
        """保存结果到文件"""
        # 保存完整结果
        with open(os.path.join(self.save_dir, 'param_search_results.json'), 'w') as f:
            json.dump({
                "best_params": self.best_params,
                "best_success_rate": self.best_success_rate,
                "all_results": self.results
            }, f, indent=4)
        
        # 保存简化版CSV结果
        with open(os.path.join(self.save_dir, 'results_summary.csv'), 'w') as f:
            f.write("attract_coef,reject_coef,contact_coef,model_param,reject_dis,attract_point_comp,success_rate,avg_steps\n")
            for result in self.results:
                params = result["params"]
                f.write(f"{params.get('attract_coef', '')},"
                        f"{params.get('reject_coef', '')},"
                        f"{params.get('contact_coef', '')},"
                        f"{params.get('model_param', '')},"
                        f"{params.get('reject_dis', '')},"
                        f"{params.get('attract_point_comp', '')},"
                        f"{params.get('ori_coef', '')},"
                        f"{params.get('contact_cost_param', '')},"
                        f"{result['success_rate']},"
                        f"{np.mean([t['steps'] for t in result['trials'] if 'steps' in t])}\n")

if __name__ == "__main__":
    # 基础参数配置
    base_args = Namespace(
        obj='teapot',
        attract_coef=0.5,
        reject_coef=0.001,
        contact_coef=0.5,
        contact_cost_param=0,
        model_param=15,
        reject_dis=0.01,
        attract_point_comp=0.1,
        ground_height_threshold=0.012,
        sample_num=70,
        pos_coef=1,
        ori_coef=0.0005,
        low_err_coef=0.3,
        upper_err_coef=1
    )
    
    # 创建并运行优化器
    optimizer = HyperparameterOptimizer(
        base_args,
        num_trials=100,
        max_workers=8,  # 可以根据CPU核心数调整
        max_param_num=1000
    )
    optimizer.run_optimization()
