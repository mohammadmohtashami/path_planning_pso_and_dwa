import numpy as np
import matplotlib.pyplot as plt

# --- 1. Map Definition ---
map_size = 100
obstacles = [
    # (x, y, width, height)
    (20, 20, 10, 30), (60, 40, 10, 30), (20, 60, 30, 10),
    (60, 70, 10, 30), (40, 30, 30, 10), (60, 0, 10, 20), (90, 55, 15, 10)
]
start_point = np.array([10, 90])
target_point = np.array([95, 90])

# --- 2. Advanced PSO Parameters ---
num_waypoints = 5      # 5 نقطه کافی است
num_particles = 400    # تعداد زیاد برای پیدا کردن سوراخ موش!
max_iter = 300
w_max, w_min = 0.9, 0.2
c1, c2 = 2.0, 1.5      # تمرکز بیشتر روی تجربه شخصی (Exploration)

def get_rect_intersection_length(p1, p2, rect):
    """
    محاسبه دقیق ریاضی: طول بخشی از خط که داخل مستطیل است.
    بدون نیاز به نقاط تست، کاملاً دقیق.
    """
    rx, ry, rw, rh = rect
    rx_max, ry_max = rx + rw, ry + rh
    
    # بردار جهت خط
    d = p2 - p1
    if np.all(d == 0): return 0.0
    
    # پارامترهای t برای ورود و خروج به باکس در محور X
    # p1[0] + t * d[0] = rx  --> t = (rx - p1[0]) / d[0]
    t_min, t_max = 0.0, 1.0
    
    # بررسی محور X
    if abs(d[0]) < 1e-9: # خط عمودی
        if p1[0] < rx or p1[0] > rx_max: return 0.0
    else:
        t1 = (rx - p1[0]) / d[0]
        t2 = (rx_max - p1[0]) / d[0]
        t_enter_x = min(t1, t2)
        t_exit_x = max(t1, t2)
        t_min = max(t_min, t_enter_x)
        t_max = min(t_max, t_exit_x)
        
    # بررسی محور Y
    if abs(d[1]) < 1e-9: # خط افقی
        if p1[1] < ry or p1[1] > ry_max: return 0.0
    else:
        t1 = (ry - p1[1]) / d[1]
        t2 = (ry_max - p1[1]) / d[1]
        t_enter_y = min(t1, t2)
        t_exit_y = max(t1, t2)
        t_min = max(t_min, t_enter_y)
        t_max = min(t_max, t_exit_y)
        
    # اگر بازه زمانی معتبر بود، یعنی برخورد داریم
    if t_min < t_max:
        # طول برخورد = نسبت زمان در برخورد * طول کل خط
        return (t_max - t_min) * np.linalg.norm(d)
    return 0.0

def calculate_fitness(path_waypoints):
    full_path = np.vstack([start_point, path_waypoints.reshape(-1, 2), target_point])
    total_dist = 0
    total_intersection = 0
    
    # جریمه خروج از محدوده نقشه
    out_of_bounds = np.sum(full_path < 0) + np.sum(full_path > map_size)
    
    for i in range(len(full_path) - 1):
        p1, p2 = full_path[i], full_path[i+1]
        total_dist += np.linalg.norm(p1 - p2)
        
        for obs in obstacles:
            # محاسبه دقیق طول نفوذ در دیوار
            total_intersection += get_rect_intersection_length(p1, p2, obs)

    # فرمول نهایی: اولویت مطلق با رفع برخورد است
    # ضریب 10,000 یعنی: 1 واحد نفوذ در دیوار = 10,000 واحد جریمه
    penalty = (total_intersection * 10000) + (out_of_bounds * 50000)
    return total_dist + penalty

# --- 3. Initialization ---
dim = num_waypoints * 2
particles = np.random.uniform(0, map_size, (num_particles, dim))
velocity = np.random.uniform(-5, 5, (num_particles, dim))

p_best = particles.copy()
p_best_fitness = np.array([calculate_fitness(p) for p in particles])
g_best = p_best[np.argmin(p_best_fitness)].copy()
g_best_fitness = np.min(p_best_fitness)

print("Searching for the gap...")

# --- 4. Main Loop ---
for it in range(max_iter):
    w = w_max - (it / max_iter) * (w_max - w_min)
    
    # هر 50 دور، اگر هنوز مسیر بسته‌ایم، ذرات را شوک بده
    if it % 50 == 0 and g_best_fitness > 5000:
        # شوک: پرت کردن نیمی از ذرات به نقاط کاملاً تصادفی جدید
        mask = np.random.choice([True, False], num_particles)
        particles[mask] = np.random.uniform(0, map_size, (np.sum(mask), dim))
        velocity[mask] = np.random.uniform(-10, 10, (np.sum(mask), dim))
        # ریست کردن حافظه شخصی آن‌ها
        p_best[mask] = particles[mask]
        p_best_fitness[mask] = np.array([calculate_fitness(p) for p in particles[mask]])
    
    for i in range(num_particles):
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        velocity[i] = (w * velocity[i] + 
                       c1 * r1 * (p_best[i] - particles[i]) + 
                       c2 * r2 * (g_best - particles[i]))
        
        velocity[i] = np.clip(velocity[i], -8, 8)
        particles[i] += velocity[i]
        particles[i] = np.clip(particles[i], 0, map_size)
        
        fit = calculate_fitness(particles[i])
        if fit < p_best_fitness[i]:
            p_best[i] = particles[i].copy()
            p_best_fitness[i] = fit
            
    # Update Global Best
    min_idx = np.argmin(p_best_fitness)
    if p_best_fitness[min_idx] < g_best_fitness:
        g_best = p_best[min_idx].copy()
        g_best_fitness = p_best_fitness[min_idx]

# --- 5. Visualization ---
best_path = np.vstack([start_point, g_best.reshape(-1, 2), target_point])
plt.figure(figsize=(10, 10))

# رسم دقیق موانع
for (ox, oy, w, h) in obstacles:
    plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color='salmon', ec='darkred', alpha=0.8))

# رسم مسیر
color = 'green' if g_best_fitness < 2000 else 'red'
plt.plot(best_path[:, 0], best_path[:, 1], '-o', color=color, linewidth=2, label='PSO Path')
plt.scatter([start_point[0], target_point[0]], [start_point[1], target_point[1]], c='blue', s=150, zorder=5)

plt.xlim(0, 100); plt.ylim(0, 100)
plt.title(f"Fitness: {g_best_fitness:.1f}\n(Must be < 2000 for success)")
plt.grid(True)
plt.legend()
plt.show()