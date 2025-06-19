import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import minimize

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path)
    time = df['time'].values
    u = df['volte'].values
    y = df['temperature'].values
    
    y_initial = np.mean(y[:50])
    step_idx = np.argmax(u > u[0] * 1.1)
    t_step = time[step_idx:]
    
    return {
        'time': time,
        'u': u,
        'y': y,
        't_step': t_step - t_step[0],  # 以阶跃时刻为时间起点
        'y_step': y[step_idx:],        # 保持与时间同步的截断
        'y_initial': y_initial
    }

def identify_system_parameters(data):
    """系统参数辨识函数"""
    initial_temp = data['y_initial']
    steady_data = data['y_step'][-int(len(data['y_step'])*0.1):]
    final_temp = np.mean(steady_data)
    step_voltage = data['u'][0]
    
    # 计算系统增益
    system_gain = (final_temp - initial_temp) / step_voltage
    
    # 确定延迟时间（使用5%阈值）
    threshold_temp = initial_temp + 0.05 * (final_temp - initial_temp)
    start_index = np.argmax(data['y_step'] > threshold_temp)
    delay_time = data['t_step'][start_index] if start_index < len(data['t_step']) else 0
    
    # 计算时间常数（使用28%和63%特征点）
    y28 = initial_temp + 0.28 * (final_temp - initial_temp)
    y63 = initial_temp + 0.63 * (final_temp - initial_temp)
    
    idx_28 = (np.abs(data['y_step'] - y28)).argmin()
    idx_63 = (np.abs(data['y_step'] - y63)).argmin()
    
    t28 = data['t_step'][idx_28] - delay_time
    t63 = data['t_step'][idx_63] - delay_time
    
    time_constant = (t63 - t28) / np.log((1-0.28)/(1-0.63))
    
    return {
        'K': system_gain,
        'L': delay_time,
        'T': time_constant
    }

def verify_model(params, data):
    """验证辨识模型"""
    t_sim = data['time']
    y_initial = data['y_initial']
    K = params['K']
    L = params['L']
    T = params['T']
    u = data['u']
    
    num = [K]
    den = [T, 1]
    system = signal.TransferFunction(num, den)
    t_out, y_out_no_delay = signal.step(system, T=t_sim)
    
    delta_u = u.max() - 0
    y_out_no_delay = y_initial + delta_u * y_out_no_delay
    
    dt = t_sim[1] - t_sim[0]
    delay_steps = int(L / dt)
    y_out = np.concatenate([np.ones(delay_steps) * y_initial, y_out_no_delay[:-delay_steps]])
    
    return {
        't_sim': t_sim,
        'y_out': y_out,
        'y_out_no_delay': y_out_no_delay
    }

def optimize_pid_params(data, model_params):
    """优化PID参数"""
    K_process = model_params['K']
    L_process = model_params['L']
    T_process = model_params['T']
    
    t_sim = np.linspace(0, 10000, 10000)
    setpoint = 35.0
    
    def pid_sim(Kp, Ki, Kd):
        num_pid = [Kd, Kp, Ki]
        den_pid = [1, 0]
        num_process = [K_process]
        den_process = [T_process, 1]
        
        num_open = np.polymul(num_pid, num_process)
        den_open = np.polymul(den_pid, den_process)
        
        num_closed = num_open
        den_closed = np.polyadd(den_open, num_open)
        
        closed_loop = signal.TransferFunction(num_closed, den_closed)
        t_out, y_out_no_delay = signal.step(closed_loop, T=t_sim)
        
        dt = t_sim[1] - t_sim[0]
        delay_steps = int(L_process / dt)
        y_out = np.concatenate([np.ones(delay_steps) * y_out_no_delay[0], y_out_no_delay[:-delay_steps]])
        y_out *= setpoint
        
        return t_out, y_out
    
    def performance_index(params):
        Kp, Ki, Kd = params
        t_out, y_out = pid_sim(Kp, Ki, Kd)
        error = y_out - setpoint
        return np.sum(error ** 2)
    
    initial_attempt = [5.0, 0.5, 2.5]

    bounds = [(0.1, 200.0), (0.05, 100.0), (0.05, 50.0)]  # Kp, Ki, Kd的边界

    result = minimize(
        performance_index,
        initial_attempt,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': False, 'maxiter': 300}
    )
    
    best_params = result.x
    Kp_opt, Ki_opt, Kd_opt = best_params
    t_out, y_out = pid_sim(Kp_opt, Ki_opt, Kd_opt)
    
    return {
        'Kp': Kp_opt,
        'Ki': Ki_opt,
        'Kd': Kd_opt,
        't_out': t_out,
        'y_out': y_out
    }

def calculate_performance(t_out, y_out, setpoint, tolerance=0.02):
    """计算性能指标"""
    y_5 = setpoint * 0.05
    y_95 = setpoint * 0.95
    
    try:
        t_rise_start = t_out[np.where(y_out >= y_5)[0][0]]
        t_rise_end = t_out[np.where(y_out >= y_95)[0][0]]
        rise_time = t_rise_end - t_rise_start
    except:
        rise_time = np.nan
    
    max_temp = np.max(y_out)
    overshoot = ((max_temp - setpoint) / setpoint) * 100
    
    y_steady_state = np.mean(y_out[-100:])
    steady_state_error = abs(setpoint - y_steady_state)
    
    lower_bound = setpoint * (1 - tolerance)
    upper_bound = setpoint * (1 + tolerance)
    settling_time = np.nan
    
    for i in range(len(y_out)):
        if np.all((y_out[i:] >= lower_bound) & (y_out[i:] <= upper_bound)):
            settling_time = t_out[i]
            break
    
    return {
        "Rise Time (s)": rise_time,
        "Overshoot (%)": overshoot,
        "Steady-State Error (°C)": steady_state_error,
        "Settling Time (s)": settling_time
    }

def visualize_results(data, model_params, verification, optimized_pid, metrics, setpoint=35):
    """可视化结果"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    
    # 创建模型验证图
    plt.figure(figsize=(12, 6))
    
    # 实际温度曲线（蓝色实线）
    plt.plot(data['time'], data['y'], 'b-', label='实际温度', linewidth=1.5)

    # 模型预测曲线（黑色虚线）
    plt.plot(verification['t_sim'], verification['y_out'], 'k--', 
            label='FOPDT模型预测', linewidth=1.5)
    
    # 图表样式设置
    plt.xlabel('时间 (s)')
    plt.ylabel('温度 (°C)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 参数显示框
    text_box = f"""
    系统辨识参数:
    -------------------------
    初始温度: {model_params['K']:.2f}°C
    稳态温度: {np.mean(data['y'][-int(len(data['y'])*0.1):]):.2f}°C
    系统增益: {model_params['K']:.4f} °C/V
    时间常数: {model_params['T']:.2f} s
    延迟时间: {model_params['L']:.2f} s
    
    性能指标:
    -------------------------
    上升时间 (s)     |{metrics['Rise Time (s)']:.2f}
    调节时间 (s)     |{metrics['Settling Time (s)']:.2f}
    """
    
    plt.figtext(0.65, 0.15, text_box, fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig('system_identification.png', dpi=300)
    plt.close()
    
    # 创建PID优化结果图
    plt.figure(figsize=(12, 6))
    
    # 优化后的PID响应曲线（红色实线）
    plt.plot(optimized_pid['t_out'], optimized_pid['y_out'], 'r-', 
            label='优化后的温度', linewidth=1.5)
    # 设定温度线（绿色虚线）
    plt.plot(optimized_pid['t_out'], np.ones_like(optimized_pid['t_out']) * setpoint, 
            'g--', label=f'设定温度 ({setpoint}°C)', linewidth=0.5)
    
    # 图表样式设置
    plt.xlabel('时间 (s)')
    plt.ylabel('温度 (°C)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # PID参数显示框
    pid_text = f"""
    优化后的PID参数:
    -------------------------
    比例增益 (Kp): {optimized_pid['Kp']:.4f}
    积分增益 (Ki): {optimized_pid['Ki']:.4f}
    微分增益 (Kd): {optimized_pid['Kd']:.4f}
    """
    
    plt.figtext(0.7, 0.2, pid_text, fontsize=9, 
               bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    plt.savefig('pid_optimization_improved.png', dpi=300)
    plt.close()

def create_report(model_params, metrics, optimized_pid):
    """创建文本报告"""
    report = f"""系统辨识与PID控制分析报告
==========================================

系统辨识结果:
------------------------------------------
比例系数 K: {model_params['K']:.4f} °C/V
延迟时间 L: {model_params['L']:.2f}s
时间常数 T: {model_params['T']:.2f}s

模型方程:
------------------------------------------
辨识模型: G(s) = {model_params['K']:.4f}/({model_params['T']:.2f}s+1)·e^(-{model_params['L']:.2f}s)

性能指标:
------------------------------------------
上升时间 (s)     {metrics['Rise Time (s)']:.2f}
超调量 (%)       {metrics['Overshoot (%)']:.2f}
稳态误差 (°C)    {metrics['Steady-State Error (°C)']:.2f}
调节时间 (s)     {metrics['Settling Time (s)']:.2f}

优化后的PID参数:
------------------------------------------
比例增益 (Kp): {optimized_pid['Kp']:.4f}
积分增益 (Ki): {optimized_pid['Ki']:.4f}
微分增益 (Kd): {optimized_pid['Kd']:.4f}
==========================================
"""
    
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """主函数"""
    # 数据加载与预处理
    data = load_and_preprocess_data("temperature.csv")
    
    # 系统辨识
    model_params = identify_system_parameters(data)
    
    # 模型验证
    verification = verify_model(model_params, data)
    
    # PID参数优化
    optimized_pid = optimize_pid_params(data, model_params)
    
    # 性能评估
    metrics = calculate_performance(
        optimized_pid['t_out'], 
        optimized_pid['y_out'], 
        setpoint=35
    )
    
    # 可视化结果
    visualize_results(data, model_params, verification, optimized_pid, metrics, setpoint=35)
    
    # 创建文本报告
    create_report(model_params, metrics, optimized_pid)
    
    print("分析完成！结果已保存到 analysis_report.txt、system_identification.png 和 pid_optimization_improved.png")

if __name__ == "__main__":
    main()