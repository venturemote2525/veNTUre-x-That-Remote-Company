export interface Device {
  mac: string;
  name?: string;
  bindStatus?: number;
}

export interface ICDevice {
  mac: string;
  name?: string;
  rssi?: number;
  isConnected?: boolean;
}

export interface ICWeightData {
  weight: number;
  timestamp: number;
  impedance?: number;
  isStabilized?: boolean;
  unit?: string;

  [key: string]: any;
}

export interface ICWeightMeasurement {
  device: ICDevice;
  data: ICWeightData;
  timestamp: number;
}

export interface ICDeviceInfo {
  firmwareVersion?: string;
  hardwareVersion?: string;

  [key: string]: any;
}

export interface ICUserInfo {
  name: string;
  gender: 'FEMALE' | 'MALE';
  age: number;
  height: number;
}
