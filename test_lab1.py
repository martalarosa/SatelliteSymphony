import pytest
import os
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # отключаем открытие окон при тестах
import matplotlib.pyplot as plt

from SatelliteSymphony import SP3Parser, RinexParser, OrbitSimulator, EphemerisPlot

# путь к тестовым данным
SP3_TEST_FILE = "Sta22440.sp3"
RINEX_TEST_FILE = "Brdc2380.24n"

@pytest.mark.skipif(not Path(SP3_TEST_FILE).exists(), reason="SP3 файл не найден")
def test_parse_sp3_valid():
    parser = SP3Parser(SP3_TEST_FILE)
    data = parser.parse()
    assert isinstance(data, list)
    assert len(data) > 0
    # проверим структуру записи
    sample = data[0]
    assert 'sat' in sample and 'time' in sample and 'coords' in sample
    assert isinstance(sample['coords'], list) and len(sample['coords']) == 3

@pytest.mark.skipif(not Path(RINEX_TEST_FILE).exists(), reason="RINEX файл не найден")
def test_parse_rinex_valid():
    parser = RinexParser(RINEX_TEST_FILE)
    data = parser.parse()
    assert isinstance(data, list)
    assert len(data) > 0
    # проверим, что хотя бы один спутник – GPS, Galileo или ГЛОНАСС
    sats = {entry['sat'][0] for entry in data}
    assert any(s in ['G', 'R', 'E', 'C'] for s in sats)

def test_orbit_simulation_generation():
    sim = OrbitSimulator()
    data = sim.generate(count=120)
    assert isinstance(data, list)
    assert len(data) == 120
    assert all('sat' in d and 'coords' in d and len(d['coords']) == 3 for d in data)

def test_plot_ephemeris_filtering():
    from datetime import datetime, timedelta
    now = datetime.now()
    data = [{'sat': 'TST', 'time': now, 'coords': [1, 2, 3]},
            {'sat': 'TST', 'time': now + timedelta(minutes=5), 'coords': [4, 5, 6]}]
    plot = EphemerisPlot(data)
    try:
        plot.filter_and_plot('TST')  # построит, но не покажет
        plt.close('all')  # закрываем фигуры
    except Exception as e:
        pytest.fail(f"Plotting вызвал исключение: {e}")

def test_empty_data_handling():
    parser = SP3Parser("nonexistent_file.sp3")
    with pytest.raises(FileNotFoundError):
        parser.parse()
