#!/usr/bin/env python3
from argparse import ArgumentParser
from pythonosc import udp_client
import serial
import time
import numpy as np
from typing import Tuple, Union, Optional
from matplotlib import pyplot as plt
from biosppy.signals import ecg
from scipy import signal
import threading
from copy import copy

def main():
    args = getargs()

    history_buf = HistoryBuffer(
        serial_port = args.serial_port,
        serial_speed = args.serial_speed,
        fs_supposed = args.fs_supposed,
        fs_torelance = args.fs_torelance,
        history_buf_size = args.history_buf_size,
        bpm_estim_cycletime_sec = args.bpm_estim_cycletime_sec,
        sensor_value_thres = args.sensor_value_thres,
    )

    thread_updatebuf = threading.Thread(target=history_buf.update_repeat)
    thread_calculation = threading.Thread(target=history_buf.calc_repeat)

    thread_updatebuf.start()
    thread_calculation.start()


class HistoryBuffer():
    """
    Class for buffering and processing incoming sensor data from serial port.
    For calculating BPM from buffered data, this class calls BeatAnalyzer.analyze() method.
    Buffer update cycletime depends on sensor sampling rate; if sensor send data every 1ms, __receive_onedata() method wait for and receive data every 1ms.
    """
    def __init__(
        self,
        serial_port: str,
        serial_speed: int = 115200,
        fs_supposed: int = 200,
        fs_torelance: int = 2,
        history_buf_size: int = 800,
        bpm_estim_cycletime_sec: float = 1.0,
        sensor_value_thres: int = 50000,
    ):
        self.ser = serial.Serial(serial_port, serial_speed, timeout=0.1) 
        self.history_buf_size = history_buf_size
        self.bpm_estim_cycletime_sec = bpm_estim_cycletime_sec
        self.fs_supposed = fs_supposed
        self.fs_torelance = fs_torelance
        self.sensor_value_thres = sensor_value_thres

        self.data = {
            "time": np.zeros([history_buf_size]),
            "value": np.zeros([history_buf_size]),
        }

        self.is_buf_full = False
        self.datacnt = 0

        self.beatanalyzer = BeatAnalyzer(fs = fs_supposed, Nsamples=history_buf_size)

    ################### data aquisition related #################
    def update_repeat(self):
        while True:
            self.__update()

    def __update(self):
        # get incoming data
        sensor_value, sensor_time = self.__receive_onedata()
        if sensor_value is not None:
            # update ring buffer
            self.data["value"] = np.roll(self.data["value"], 1)
            self.data["value"][0] = sensor_value
            self.data["time"] = np.roll(self.data["time"], 1)
            self.data["time"][0] = sensor_time
            
            # update count
            if not self.is_buf_full and self.data["time"][-1] > 0:
                self.is_buf_full = True

    def __receive_onedata(self) -> Tuple[Union[int, None], Union[float, None]]:
        buf = ''
        while True:
            readtime = time.time()
            data = self.ser.read_all()

            if len(data) > 0: 
                # if something is received
                try:
                    data_str = data.decode()
                except:
                    print(f'skip {data}')
                    return None, None
                
                buf += data_str

                if '\n' == data_str[-1]:
                    # if data completed
                    buf_datas = buf.split('\r\n')
                    last_buf_data = buf_datas[-2] # because buf_datas is typically like [..., '1214', '1213', '']
                    
                    # buffed data can somhow be incomplete
                    try:
                        sensor_value = int(last_buf_data)
                    except:
                        print(f'skip {buf}')
                        return None, None

                    sensor_time = time.time()
                    break
            
            time.sleep(0.1/1000)

        return sensor_value, sensor_time

    ################### bpm calculation related #################
    def calc_repeat(self):
        while True:
            self.__calc_bpm()
            time.sleep(self.bpm_estim_cycletime_sec)

    def __calc_bpm(self):
        # get buffered data
        buf_data = copy(self.data['value'])
        buf_time = copy(self.data['time'])

        if not self.is_buf_full:
            return

        # calculate measured fs based on data aquisition time
        fs_actual = self.history_buf_size / (buf_time[0] - buf_time[-1])
        if abs(fs_actual - self.fs_supposed) > self.fs_torelance:
            print(f"Measured fs: {fs_actual} is too different from fs: {self.fs_supposed}")
            return
        
        # if any of bufferred sensor value are less than threshold, skip culculation. It seems nothing is attached to the sensor.
        if np.min(self.data["value"]) < self.sensor_value_thres:
            print(f"Min bufferred sensor value: {np.min(buf_data)} indicates nothing is attached to the sensor.")
            return

        ### ecg toolkit based bpm estimation
        # analyzed = ecg.ecg(signal=self.data["value"][::-1], sampling_rate=fs_actual, show=False)
        # estim_bpm = np.median(analyzed["heart_rate"])
        ### custom bpm estimation
        estim_bpm = self.beatanalyzer.analyze(buf_data[::-1], actual_fs = fs_actual)

        print(f"estimated bpm: {estim_bpm:.01f}")

class BeatAnalyzer():
    """
    Class for calculating beat from time series sensor data.
    """
    def __init__(
        self,
        fs: int,
        Nsamples: int,
        min_bpm: float = 40.0,
        max_bpm: float = 180.0,
    ):
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.fs = fs
        self.Nsamples = Nsamples
        self.datalen_sec = Nsamples/fs
        self.fft_bpm_resolution = fs/Nsamples*60
        self.bpms = np.arange(np.ceil(min_bpm/self.fft_bpm_resolution), np.floor(max_bpm/self.fft_bpm_resolution)+1) * self.fft_bpm_resolution
        basis_phase = 2 * np.pi / fs * np.dot(self.bpms[:,None]/60, np.arange(0,Nsamples)[None,:])
        self.basis_sin = np.sin(basis_phase)
        self.basis_cos = np.cos(basis_phase)

    def analyze(self, data: np.ndarray, actual_fs: float) -> float:
        # step1 detect rough bpm based on fft
        cor = np.sqrt((np.dot(self.basis_sin, data)/self.Nsamples) ** 2 + (np.dot(self.basis_cos, data)/self.Nsamples) ** 2)
        rough_bpm = self.bpms[np.argmax(cor)]
        print(f"rough_bpm: {rough_bpm}")
        rough_fs_range = [max(rough_bpm/60 - 0.5, self.min_bpm/60), min(rough_bpm/60 + 0.5, self.max_bpm/60)]

        # step2 filter using rough bpm range
        filter_coef = signal.firwin(numtaps=91, cutoff=rough_fs_range, fs=actual_fs, pass_zero=False)
        filterd_signal = signal.lfilter(filter_coef, 1, data)[92:]

        # step3 detect local extremums
        extremums = np.diff((np.diff(filterd_signal) > 0).astype(int)) 
        local_maxima = np.where(extremums == -1)[0] + 1
        local_minima = np.where(extremums == 1)[0] + 1
        # print(local_maxima/200)
        # print(local_minima/200)

        # step4 check if each local maxima/minima is max/min within a certain range
        is_local_maxima_max = np.zeros(len(local_maxima))
        is_local_minima_min = np.zeros(len(local_minima))
        detect_width = int(actual_fs/(self.max_bpm/60) / 2)
        for i, idx in enumerate(local_maxima):
            target_range = [max(0, idx - detect_width), idx + detect_width]

            argmax_within_range = np.argmax(filterd_signal[target_range[0]:target_range[1]]) + idx - detect_width
            if idx == argmax_within_range:
                is_local_maxima_max[i] = 1

        for i, idx in enumerate(local_minima):
            target_range = [max(0, idx - detect_width), idx + detect_width]

            argmin_within_range = np.argmin(filterd_signal[target_range[0]:target_range[1]]) + idx - detect_width
            if idx == argmin_within_range:
                is_local_minima_min[i] = 1
        
        max_candidate = local_maxima[np.where(is_local_maxima_max == 1)[0]]
        min_candidate = local_minima[np.where(is_local_minima_min == 1)[0]]
        # print(max_candidate/200)
        # print(min_candidate/200)
        if len(max_candidate) < 2 or len(min_candidate) < 2:
            print(f"No enough pulse detected.")
            return None

        # step5 calculate period
        period_max = np.diff(max_candidate)
        period_min = np.diff(min_candidate)

        # step6 calculate average bpm
        if len(period_max) >= 3:
            period_max = period_max[1:-1]
        if len(period_min) >= 3:
            period_min = period_min[1:-1]
        period_mean = np.mean(np.hstack([period_max, period_min])/actual_fs)

        return 1/period_mean * 60

def getargs():
    argparser = ArgumentParser()
    argparser.add_argument('--serial_port', type=str, required=True, help='Arduino serial port. Specify like "COM4"')
    argparser.add_argument('--serial_speed', type=int, default=115200, help='Serial speed in bps')
    argparser.add_argument('--history_buf_size', type=int, default=800, help='Sensor data buffer size')
    argparser.add_argument('--bpm_estim_cycletime_sec', type=float, default=1.0, help='Estimate BPM using bufferred data every specified cycles')
    argparser.add_argument('--fs_supposed', type=int, default=200, help='Sampling rate of the sensor.')
    argparser.add_argument('--fs_torelance', type=int, default=2, help='If actual sample rate goes below this value, stop updating BPM since something seems wrong...')
    argparser.add_argument('--sensor_value_thres', type=int, default=50000, help='If data buf contains value less than this threshold, skip BPM estimation.')
    args = argparser.parse_args()
    return args

def int_or_none(string: str) -> Optional[int]:
    if string.strip() in ('none', 'None', 'NONE', 'null', 'Null', 'NULL'):
        return None
    return int(string)

if __name__ == '__main__':
    main()