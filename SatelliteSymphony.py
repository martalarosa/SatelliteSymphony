""" данный код реализует программно-математическое обеспечение (ПМО) для анализа траекторий навигационных искусственных спутников земли (НИСЗ). 
поддерживаются реальные данные из файлов RINEX 3.02 и SP3, а также режим имитации с периодической моделью орбиты спутника.
интерфейс (tkinter) позволяет выбрать режим, загрузить данные (либо сгенерировать орбиту), 
выбрать спутниковую систему, номер спутника и временной интервал. поддерживаются навигационные созвездия GPS, ГЛОНАСС, Galileo и BDS. 
в виде графиков отображаются эволюция координат и скоростей. реализовано покрытие  юнит- и интеграционными тестами, 
настроен pipeline для github actions и gitlab ci с автоматическим запуском тестов при каждом обновлении проекта. """

import math
import re
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

EARTH_GRAVITY = 3.986005e14       # гравитационный параметр Земли (м^3/с^2)
EARTH_ROTATION_RATE = 7.2921151467e-5  # угловая скорость вращения Земли (рад/с)

# имитационный режим
class OrbitSimulator:
    def generate(self, count=500):
        # генерируем count точек через равные интервалы (минуты), начиная от текущего времени
        start_time = datetime.now()
        times = [start_time + timedelta(seconds=60*i) for i in range(count)]
        # определяем условные амплитуды орбиты по осям
        Ax, Ay, Az = 20200.0, 14000.0, 21500.0  # км (приблизительные размеры орбиты)
        coords_list = []
        # вычисляем координаты с помощью гармонических функций для имитации орбиты
        for i in range(count):
            # моделируем периодическое изменение координат
            x = Ax * math.cos(2 * math.pi * i / count)
            y = Ay * math.sin(2 * math.pi * i / count)
            z = Az * math.cos(4 * math.pi * i / count)
            coords_list.append({'sat': 'SIM', 'time': times[i], 'coords': [x, y, z]})
        return coords_list

class RinexParser:
    # класс для парсинга навигационного RINEX файла версии 3.02
    def __init__(self, filename):
        self.filename = filename
        # будем хранить результаты парсинга эфемерид
        self.ephemerides = {}  # словарь: ключ - sat, значение - список параметров 

    def parse(self):
        data_points = []  # итоговый список точек, будет заполнен координатами после расчёта
        with open(self.filename, 'r', encoding='latin-1') as f:
            # пропускаем заголовок
            line = f.readline()
            while line and 'END OF HEADER' not in line:
                line = f.readline()
            # читаем блоки эфемерид
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            # проверяем начало нового блока (первая колонка - буква системы)
            if not line or not line[0].isalpha():
                i += 1
                continue
            sat_id = line[0:3].strip()  # первые 3 символа, например "G01"
            # парсим время эпохи (формат: YYYY MM DD hh mm ss)
            year = int(line[4:8]); month = int(line[9:11]); day = int(line[12:14])
            hour = int(line[15:17]); minute = int(line[18:20]); sec = int(line[21:23])
            epoch_time = datetime(year, month, day, hour, minute, sec)
            # определяем, сколько строк читать для данного спутника
            system = sat_id[0]  # буква системы
            if system == 'R':   # GLONASS
                num_lines = 3
            else:               # GPS, Galileo, BDS, etc.
                num_lines = 7
            # собираем все строки блока: первая уже в line, добавляем следующие num_lines
            block_lines = [line[23:]]  # начиная с 24-й позиции первой строки - первые данные
            i += 1
            for j in range(num_lines):
                if i >= len(lines): break
                # берём строку параметров, начиная с 4-х первых пробелов (данные после префикса)
                param_line = lines[i][4:].rstrip()
                block_lines.append(param_line)
                i += 1
            # объединяем строки и находим все числа в формате D (экспоненц.) через регулярку
            block_text = " ".join(block_lines)
            # заменяем символ 'D' на 'E' для возможности использования float()
            block_text = block_text.replace('D', 'E')
            values = [float(x) for x in re.findall(r'[-+]?\d+\.\d+E[-+]\d+', block_text)]
            # сохраняем параметры эфемерид
            if sat_id not in self.ephemerides:
                self.ephemerides[sat_id] = []
            self.ephemerides[sat_id].append({
                'epoch': epoch_time,
                'params': values
            })
        # после считывания всех эфемерид вычисляем координаты
        for sat, records in self.ephemerides.items():
            if len(records) == 0:
                continue
            # берём первый набор параметров
            record = records[0]
            toe = record['params'][17] if sat[0] != 'R' else 0  # Toe (сек недели) для GPS et al; для ГЛОНАСС не используем
            epoch_time = record['epoch']
            # определяем интервал времени для расчёта
            # для простоты: 12h для GPS/галилео, 3h для ГЛОНАСС (так как у него эфемериды на 30 минут обычно)
            if sat[0] == 'R':
                duration = 3 * 3600  # 3 часа
            else:
                duration = 12 * 3600  # 12 часов - примерный период GPS
            end_time = epoch_time + timedelta(seconds=duration)
            # вычисляем траекторию с шагом 300 секунд (5 минут)
            t = epoch_time
            step = 300 
            prev_coords = None
            while t <= end_time:
                # вычисляем координаты спутника на момент t
                coords = self._compute_coords_for_time(sat, record, t)
                if coords:
                    data_points.append({'sat': sat, 'time': t, 'coords': coords})
                t += timedelta(seconds=step)
        return data_points

    def _compute_coords_for_time(self, sat, record, t):
        # sспомогательная функция, рассчитывающая координаты спутника sat на времени t по заданной эфемериде record
        params = record['params']
        sys = sat[0]
        # обрабатываем отдельно GLONASS и остальные
        if sys == 'R':
            # для ГЛОНАСС: параметры содержат сразу позицию (XYZ), скорость (VxVyVz) и ускорение на эпоху
            ''' в RINEX для GLONASS формат: 
            params[0]: X (км), [1]: dX (км/с), [2]: д^2X (км/с^2), [3]: здоровье спутника
            [4]: Y, [5]: dY, [6]: д^2Y, [7]: номер частоты, [8]: Z, [9]: dZ, [10]: д^2Z, [11]: возраст данных '''
            # преобразуем км -> м, км/с -> м/с, км/с^2 -> м/с^2
            if len(params) < 11:
                return None
            # исходные данные на эпохе (t0 = record['epoch'])
            t0 = record['epoch']
            # позиции и скорости в СК исходно
            X0 = params[0] * 1000.0; Y0 = params[4] * 1000.0; Z0 = params[8] * 1000.0
            Vx0 = params[1] * 1000.0; Vy0 = params[5] * 1000.0; Vz0 = params[9] * 1000.0
            Ax0 = params[2] * 1000.0; Ay0 = params[6] * 1000.0; Az0 = params[10] * 1000.0
            # время от эпохи эфемерид
            dt = (t - t0).total_seconds()
            # интерполируем положение: r = r0 + v0*dt + 0.5*a0*dt^2
            X = X0 + Vx0*dt + 0.5*Ax0*(dt**2)
            Y = Y0 + Vy0*dt + 0.5*Ay0*(dt**2)
            Z = Z0 + Vz0*dt + 0.5*Az0*(dt**2)
            return [X/1000.0, Y/1000.0, Z/1000.0]  # переводим обратно в км для графиков
        else:
            ''' для GPS, Galileo, BDS и прочих систем с кеплеровыми параметрами:
            из массива params берем соответствующие значения по их порядку в RINEX 3.02:
            индексы (с учётом, что первая строка – 4 значения часов, дальше 7 строк по 4):
            params:
            [0]=SV clock bias; [1]=SV clock drift; [2]=SV clock drift rate
            [3]=IODE; [4]=Crs; [5]=delta n; [6]=M0;
            [7]=Cuc; [8]=e; [9]=Cus; [10]=sqrtA;
            [11]=Toe; [12]=Cic; [13]=Ω0; [14]=Cis;
            [15]=i0; [16]=Crc; [17]=ω; [18]=Ω̇;
            [19]=IDOT; ... (далее дополнительные параметры, напр. неделя, TGD, etc., которые для расчета позиций не нужны напрямую) '''
            if len(params) < 19:
                return None
            # получаем основные элементы орбиты
            delta_n   = params[5]        # изменение среднего движения (рад/с)
            M0        = params[6]        # средняя аномалия на эпоху (рад)
            e         = params[8]        # эксцентриситет
            sqrtA     = params[10]       # sqrt(a) (sqrt(м))
            Toe       = params[11]       # время эфемериды (сек GPS недели)
            Omega0    = params[13]       # долгота восходящего узла (рад)
            i0        = params[15]       # наклонение на эпоху (рад)
            Crs       = params[4]; Crc = params[16]   # радиусные поправки (м)
            Cuc       = params[7]; Cus = params[9]    # поправки к аргументу широты (рад)
            Cic       = params[12]; Cis = params[14]  # поправки к наклонению (рад)
            omega     = params[17]       # аргумент перигея (рад)
            Omega_dot = params[18]       # скорость изменения omega (рад/с)
            IDOT      = params[19]       # скорость изменения наклонения (рад/с)
            # вычисляем производные величины
            A = sqrtA ** 2               # большая полуось (м)
            n0 = math.sqrt(EARTH_GRAVITY / (A**3))    # номинальное среднее движение (рад/с)
            n = n0 + delta_n            # скорректированное среднее движение
            # определяем время в секундах GPS относительно Toe
            # t может быть в формате datetime; необходимо перевести его в секунды GPS недели
            # для упрощения считаем, что t и Toe в одном недельном интервале (не через границу недели)
            # вычисляем секунды с начала недели для t и для Toe:
            # найдём начало недели (воскресенье 00:00) для эпохи Toe:
            # (здесь можем вычислить GPS week от даты Toe, но проще – взять разницу между t и Toe)
            t_sec = (t - record['epoch']).total_seconds() + Toe  # т.к. record['epoch'] соответствует Toe времени
            # более точно стоило бы узнать GPS неделю и преобразовать, но в рамках небольшого интервала, это приемлемо
            tau = t_sec - Toe  # разница от Toe
            # учитываем периодичность: если |tau| > 302400 (3.5 дня), считаем переход через конец недели
            if tau > 302400:
                tau -= 604800
            elif tau < -302400:
                tau += 604800
            # 1. средняя аномалия M(t)
            M = M0 + n * tau
            # 2. решаем уравнение Кеплера для эксцентрической аномалии E
            E = M
            # итерационный процесс:
            for _ in range(100):
                E_prev = E
                E = M + e * math.sin(E)
                if abs(E - E_prev) < 1e-12:
                    break
            # 3. истинная аномалия nu
            sin_v = math.sqrt(1 - e**2) * math.sin(E) / (1 - e * math.cos(E))
            cos_v = (math.cos(E) - e) / (1 - e * math.cos(E))
            v = math.atan2(sin_v, cos_v)
            # 4. аргумент широты phi и поправки
            phi = v + omega
            du = Cus * math.sin(2*phi) + Cuc * math.cos(2*phi)
            dr = Crs * math.sin(2*phi) + Crc * math.cos(2*phi)
            di = Cis * math.sin(2*phi) + Cic * math.cos(2*phi)
            # 5. скорректированные параметры
            u = phi + du
            r = A * (1 - e * math.cos(E)) + dr
            i = i0 + di + IDOT * tau
            # 6. положение в плоскости орбиты
            x_orb = r * math.cos(u)
            y_orb = r * math.sin(u)
            # 7. скорректированная долгота узла
            Omega = Omega0 + (Omega_dot - EARTH_ROTATION_RATE) * tau - EARTH_ROTATION_RATE * Toe
            # 8. преобразование в ECEF координаты
            X = x_orb * math.cos(Omega) - y_orb * math.cos(i) * math.sin(Omega)
            Y = x_orb * math.sin(Omega) + y_orb * math.cos(i) * math.cos(Omega)
            Z = y_orb * math.sin(i)
            # возвращаем координаты в километрах для удобства отображения
            return [X/1000.0, Y/1000.0, Z/1000.0]

class SP3Parser:
    # класс для парсинга SP3 файлов
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        data_points = []
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        current_time = None
        sat_list = []  # для хранения списка спутников (из + строк)
        for line in lines:
            if line.startswith('+'):
                # строка списка спутников
                # формат: позиции 9- (например, "+   32   G01G02...") – читаем группы по 3 символа
                ids = [line[j:j+3].strip() for j in range(9, len(line), 3)]
                for s in ids:
                    if s and s != '0':
                        sat_list.append(s)
            if line.startswith('*'):
                # читаем метку времени эпохи. формат после '*' может содержать год, месяц, день ... 
                time_str = line[2:28].strip()  # берем 26 символов после *, убирая *
                try:
                    current_time = datetime.strptime(time_str, "%Y %m %d %H %M %S")
                except:
                    # если присутствует дробная секунда, отсекаем ее и парсим
                    try:
                        current_time = datetime.strptime(time_str.split('.')[0], "%Y %m %d %H %M %S")
                    except:
                        current_time = None
            elif line.startswith('P') and current_time:
                # строки координат. общий подход: если после 'P' идет пробел и число – это индекс спутника из списка
                sat_id_field = line[1:4].strip()
                if sat_id_field.isdigit() and sat_list:
                    # числовой индекс - берём соответствующий спутник из списка
                    idx = int(sat_id_field) - 1
                    if idx < len(sat_list):
                        sat = sat_list[idx]
                    else:
                        sat = sat_id_field  # на всякий случай, если индекс вне списка
                else:
                    sat = sat_id_field
                # извлекаем координаты XYZ из фиксированных колонок (колонки 5-18, 19-32, 33-46)
                try:
                    X = float(line[4:18])
                    Y = float(line[18:32])
                    Z = float(line[32:46])
                except:
                    continue  # пропускаем строку, если не удалось преобразовать
                # сохраняем точку (SP3 координаты в км уже, согласно спецификации)
                data_points.append({'sat': sat, 'time': current_time, 'coords': [X, Y, Z]})
        return data_points

class EphemerisPlot:
    # класс для фильтрации данных и построения графиков координат и скоростей
    def __init__(self, data):
        self.data = data  # полный список словарей {'sat':..., 'time':..., 'coords':[...]}
    
    def filter_and_plot(self, sat_id, start_time=None, end_time=None):
        # фильтруем данные по идентификатору спутника
        sat_data = [entry for entry in self.data if entry['sat'] == sat_id]
        if start_time:
            sat_data = [entry for entry in sat_data if entry['time'] >= start_time]
        if end_time:
            sat_data = [entry for entry in sat_data if entry['time'] <= end_time]
        if not sat_data:
            messagebox.showerror("Ошибка", f"Нет данных для спутника {sat_id} в указанном интервале")
            return
        # сортируем по времени
        sat_data.sort(key=lambda x: x['time'])
        times = [entry['time'] for entry in sat_data]
        coords = np.array([entry['coords'] for entry in sat_data])
        # строим графики
        plt.figure(figsize=(10, 5))
        # 2D график координат (три компоненты vs время)
        plt.subplot(1, 2, 1)
        # для визуализации всех трёх координат на одном графике
        plt.plot(times, coords[:,0], label='X')
        plt.plot(times, coords[:,1], label='Y')
        plt.plot(times, coords[:,2], label='Z')
        plt.title(f"Координаты спутника {sat_id}")
        plt.xlabel("Время")
        plt.ylabel("Координаты (км)")
        plt.grid(True)
        plt.legend()
        # 2D график скоростей (производная координат)
        # вычисляем разностные скорости между последовательными точками
        if len(times) > 1:
            # переводим время в секунды для расчета скорости
            t_sec = np.array([t.timestamp() for t in times])
            # разности координат (конечные разности)
            speeds = np.diff(coords, axis=0) / np.diff(t_sec)[:, None]
            plt.subplot(1, 2, 2)
            plt.plot(times[1:], speeds[:,0], label='Vx')
            plt.plot(times[1:], speeds[:,1], label='Vy')
            plt.plot(times[1:], speeds[:,2], label='Vz')
            plt.title(f"Скорости спутника {sat_id}")
            plt.xlabel("Время")
            plt.ylabel("Скорость (км/с)")
            plt.grid(True)
            plt.legend()
        else:
            # если только одна точка, скорость определить нельзя
            plt.subplot(1, 2, 2)
            plt.text(0.1, 0.5, "Недостаточно точек для расчёта скорости", fontsize=9)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

class GNSSApp:
   # главный класс приложения
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Анализ траекторий НИСЗ")
        self._create_widgets()
        self.window.mainloop()

    def _create_widgets(self):
        tk.Button(self.window, text="Имитация движения", command=self.run_simulation).pack(pady=5)
        tk.Button(self.window, text="Реальные данные", command=self.load_real_data).pack(pady=5)

    def run_simulation(self):
        sim_data = OrbitSimulator().generate(count=360)  # 6 часов с шагом 1 мин (360 точек)
        plotter = EphemerisPlot(sim_data)
        plotter.filter_and_plot('SIM')

    def load_real_data(self):
        file_path = filedialog.askopenfilename(title="Выберите файл RINEX навигации или SP3")
        if not file_path:
            return  # если файл не выбран, прерываем выполнение
        # спрашиваем тип файла
        file_type = simpledialog.askstring("Тип файла", "Введите тип файла: 'rinex' или 'sp3'")
        if file_type is None:
            return
        file_type = file_type.strip().lower()
        if file_type not in ['rinex', 'sp3']:
            messagebox.showinfo("Замечание", "Не указан корректный тип файла, будет попытка как SP3.")
            file_type = 'sp3'
        # выбираем соответствующий парсер
        try:
            if file_type == 'rinex':
                parser = RinexParser(file_path)
            else:
                parser = SP3Parser(file_path)
            data_points = parser.parse()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}")
            return
        if not data_points:
            messagebox.showerror("Ошибка", "Файл прочитан, но данных не получено.")
            return
        # определяем список спутников из данных
        sats = sorted({entry['sat'] for entry in data_points})
        if not sats:
            messagebox.showerror("Ошибка", "В файле нет данных о спутниках.")
            return
        # словарь с расшифровками спутниковых систем
        gnss_legend = {
            'G': 'GPS',
            'R': 'GLONASS',
            'E': 'Galileo',
            'C': 'BeiDou'
        }
        sat_list = ", ".join(sats)
        legend_text = "Примечание: G - GPS, R - GLONASS, E - Galileo, C - BeiDou"
        sat_choice = simpledialog.askstring(
            "Выбор спутника",
            "Выберите спутник по идентификатору (например: G01, R07, E12, C03):\n\n" +
            legend_text + "\n\nДоступные спутники:\n" + sat_list
        )
        # проверка на пустой ввод
        if sat_choice is None or sat_choice.strip() == "":
            return
        sat_choice = sat_choice.strip().split()[0]
        # проверяем, есть ли такой спутник
        if sat_choice not in sats:
            messagebox.showerror("Ошибка", f"Спутник {sat_choice} отсутствует в данных")
            return
        t0_str = simpledialog.askstring("Начало интервала", "Введите начало интервала (YYYY-MM-DD HH:MM:SS) или оставьте пустым:")
        t1_str = simpledialog.askstring("Конец интервала", "Введите конец интервала (YYYY-MM-DD HH:MM:SS) или оставьте пустым:")
        start_dt = end_dt = None
        try:
            if t0_str:
                start_dt = datetime.strptime(t0_str, "%Y-%m-%d %H:%M:%S")
            if t1_str:
                end_dt = datetime.strptime(t1_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            messagebox.showwarning("Внимание", "Неверный формат даты/времени, будет использован полный диапазон.")
            start_dt = None
            end_dt = None

        plotter = EphemerisPlot(data_points)
        plotter.filter_and_plot(sat_choice, start_time=start_dt, end_time=end_dt)

# полетели
if __name__ == "__main__":
    GNSSApp()
