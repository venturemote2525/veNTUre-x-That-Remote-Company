import { createContext, useContext, useEffect, useState, useRef } from 'react';
import {
  NativeModules,
  NativeEventEmitter,
  Alert,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import {
  ICDevice,
  ICWeightData,
  ICWeightMeasurement,
  ICDeviceInfo,
  ICUserInfo,
} from '@/types/icdevice-types';
import { UserProfile } from '@/types/database-types';
import { useAuth } from '@/context/AuthContext';
import { calculateAge } from '@/utils/helpers';

const { ICDeviceModule } = NativeModules;
const emitter = ICDeviceModule ? new NativeEventEmitter(ICDeviceModule) : null;

interface ICDeviceContextType {
  scannedDevices: ICDevice[];
  connectedDevices: ICDevice[];
  weightData: ICWeightMeasurement[];
  isScanning: boolean;
  isSDKInitialized: boolean;
  bleState: string;
  deviceBatteryLevels: Record<string, number>;
  deviceInfo: Record<string, ICDeviceInfo>;
  bleEnabled: boolean;

  // Weight data methods
  getLatestWeightForDevice: (deviceMac: string) => ICWeightMeasurement | null;
  clearWeightData: () => void;

  // Device management methods
  initializeSDK: (profile: UserProfile) => Promise<void>;
  connectDevice: (macAddress: string) => Promise<void>;
  disconnectDevice: (macAddress: string) => Promise<void>;
  startScan: () => Promise<void>;
  stopScan: () => Promise<void>;
  clearScannedDevices: () => Promise<void>;
  refreshDevices: () => Promise<void>;
  isDeviceConnected: (macAddress: string) => Promise<boolean>;

  // Utility methods
  getBodyFatAlgorithmsManager: () => any;

  // Settings
  setUserInfo: (macAddress: string, userInfo: ICUserInfo) => Promise<void>;
}

const ICDeviceContext = createContext<ICDeviceContextType>({
  scannedDevices: [],
  connectedDevices: [],
  weightData: [],
  isScanning: false,
  isSDKInitialized: false,
  bleState: 'Unknown',
  deviceBatteryLevels: {},
  deviceInfo: {},
  bleEnabled: false,

  getLatestWeightForDevice: () => null,
  clearWeightData: () => {},

  initializeSDK: async () => {},
  connectDevice: async () => {},
  disconnectDevice: async () => {},
  startScan: async () => {},
  stopScan: async () => {},
  clearScannedDevices: async () => {},
  refreshDevices: async () => {},
  isDeviceConnected: async () => false,

  getBodyFatAlgorithmsManager: () => null,

  setUserInfo: async () => {},
});

export function ICDeviceProvider({ children }: { children: React.ReactNode }) {
  const { profile } = useAuth();
  const [scannedDevices, setScannedDevices] = useState<ICDevice[]>([]);
  const [connectedDevices, setConnectedDevices] = useState<ICDevice[]>([]);
  const [weightData, setWeightData] = useState<ICWeightMeasurement[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [isSDKInitialized, setIsSDKInitialized] = useState(false);
  const [bleState, setBleState] = useState('Unknown');
  const [deviceBatteryLevels, setDeviceBatteryLevels] = useState<
    Record<string, number>
  >({});
  const [deviceInfo, setDeviceInfo] = useState<Record<string, ICDeviceInfo>>(
    {},
  );
  const [bleEnabled, setBleEnabled] = useState(false);

  // Use ref to track initialization to prevent multiple calls
  const initializationRef = useRef<boolean>(false);
  const scanTimeoutRef = useRef<number | null>(null);

  // Get latest weight measurement for a specific device
  const getLatestWeightForDevice = (
    deviceMac: string,
  ): ICWeightMeasurement | null => {
    const deviceMeasurements = weightData
      .filter(measurement => measurement.device?.mac === deviceMac)
      .sort((a, b) => b.timestamp - a.timestamp);

    return deviceMeasurements.length > 0 ? deviceMeasurements[0] : null;
  };

  // Clear all weight data
  const clearWeightData = () => {
    setWeightData([]);
  };

  // Initialize SDK
  const initializeSDK = async (profile: UserProfile): Promise<void> => {
    if (!ICDeviceModule) {
      throw new Error('ICDeviceModule not available');
    }

    if (initializationRef.current) {
      console.log('SDK initialization already in progress or completed');
      return;
    }

    console.log('Initializing SDK...');
    try {
      const gender = profile.gender === 'FEMALE' ? 'FEMALE' : 'MALE';
      const userInfo: ICUserInfo = {
        name: profile.username,
        age: calculateAge(profile.dob),
        height: profile.height,
        gender: gender,
      };
      initializationRef.current = true;
      console.log('Initializing ICDevice SDK...');
      await ICDeviceModule.initializeSDK(userInfo);
    } catch (error) {
      initializationRef.current = false;
      console.error('Failed to initialize SDK:', error);
      throw error;
    }
  };

  // Connect to a specific device
  const connectDevice = async (macAddress: string): Promise<void> => {
    if (!ICDeviceModule) {
      throw new Error('ICDeviceModule not available');
    }

    if (!isSDKInitialized) {
      throw new Error('SDK not initialized');
    }

    try {
      console.log('Attempting to connect to device:', macAddress);
      await ICDeviceModule.connectDevice(macAddress);
    } catch (error) {
      console.error('Failed to connect to device:', error);
      throw error;
    }
  };

  // Disconnect from a specific device
  const disconnectDevice = async (macAddress: string): Promise<void> => {
    if (!ICDeviceModule) {
      throw new Error('ICDeviceModule not available');
    }

    try {
      console.log('Attempting to disconnect from device:', macAddress);
      await ICDeviceModule.disconnectDevice(macAddress);
    } catch (error) {
      console.error('Failed to disconnect from device:', error);
      throw error;
    }
  };

  // Start scanning for devices
  const startScan = async (): Promise<void> => {
    if (!ICDeviceModule) {
      throw new Error('ICDeviceModule not available');
    }

    if (!isSDKInitialized) {
      throw new Error('SDK not initialized');
    }

    try {
      console.log('Starting device scan...');
      await ICDeviceModule.startScan();
      setIsScanning(true);

      // Clear any existing timeout
      if (scanTimeoutRef.current) {
        clearTimeout(scanTimeoutRef.current);
      }

      // Set timeout to auto-stop scanning after 30 seconds
      scanTimeoutRef.current = setTimeout(async () => {
        await stopScan();
        console.log('Scan auto-stopped after 30 seconds');
      }, 30000);
    } catch (error) {
      console.error('Failed to start scan:', error);
      setIsScanning(false);
      throw error;
    }
  };

  // Stop scanning for devices
  const stopScan = async (): Promise<void> => {
    if (!ICDeviceModule) {
      throw new Error('ICDeviceModule not available');
    }

    try {
      console.log('Stopping device scan...');
      await ICDeviceModule.stopScan();
      setIsScanning(false);

      // Clear timeout
      if (scanTimeoutRef.current) {
        clearTimeout(scanTimeoutRef.current);
        scanTimeoutRef.current = null;
      }
    } catch (error) {
      console.error('Failed to stop scan:', error);
      throw error;
    }
  };

  // Clear scanned devices list
  const clearScannedDevices = async (): Promise<void> => {
    if (!ICDeviceModule) {
      throw new Error('ICDeviceModule not available');
    }

    try {
      await ICDeviceModule.clearScannedDevices();
      setScannedDevices([]);
    } catch (error) {
      console.error('Failed to clear scanned devices:', error);
      throw error;
    }
  };

  // Refresh device lists
  const refreshDevices = async (): Promise<void> => {
    if (!ICDeviceModule) {
      return;
    }

    try {
      const [connectedDevs, scannedDevs] = await Promise.all([
        ICDeviceModule.getConnectedDevices(),
        ICDeviceModule.getScannedDevices(),
      ]);

      if (connectedDevs) setConnectedDevices(connectedDevs);
      if (scannedDevs) setScannedDevices(scannedDevs);
    } catch (error) {
      console.error('Failed to refresh devices:', error);
    }
  };

  // Check if device is connected
  const isDeviceConnected = async (macAddress: string): Promise<boolean> => {
    if (!ICDeviceModule) {
      return false;
    }

    try {
      return await ICDeviceModule.isDeviceConnected(macAddress);
    } catch (error) {
      console.error('Failed to check device connection status:', error);
      return false;
    }
  };

  // Get body fat algorithms manager
  const getBodyFatAlgorithmsManager = () => {
    if (ICDeviceModule) {
      return ICDeviceModule.getBodyFatAlgorithmsManager();
    }
    return null;
  };

  const setUserInfo = async (
    macAddress: string,
    userInfo: ICUserInfo,
  ): Promise<void> => {
    if (!ICDeviceModule) throw new Error('ICDeviceModule not available');

    try {
      await ICDeviceModule.setUserInfo(macAddress, userInfo);
      console.log(`setUserInfo success for device ${macAddress}`);
    } catch (error) {
      console.error(`setUserInfo failed for device ${macAddress}:`, error);
      throw error;
    }
  };

  const setCurrentUserInfo = async (
    macAddress: string,
    userInfo: ICUserInfo,
  ): Promise<void> => {
    if (!ICDeviceModule) throw new Error('ICDeviceModule not available');

    try {
      await ICDeviceModule.updateUserInfo_W(macAddress, userInfo);
      console.log(`updateUserInfo_W success for device ${macAddress}`);
    } catch (error) {
      console.error(`updateUserInfo_W failed for device ${macAddress}:`, error);
      throw error;
    }
  };

  useEffect(() => {
    if (!ICDeviceModule || !emitter) {
      console.warn('ICDeviceModule or emitter not available');
      return;
    }

    const requestPermissions = async () => {
      if (Platform.OS === 'android' && Platform.Version >= 31) {
        const granted = await PermissionsAndroid.requestMultiple([
          PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
          PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        ]);
        console.log('Bluetooth permissions:', granted);
      }
    };

    requestPermissions();

    // Auto-initialize SDK on mount
    const autoInitialize = async (profile: UserProfile) => {
      if (!initializationRef.current && !isSDKInitialized) {
        try {
          console.log('Auto initialise');
          await initializeSDK(profile);
          setIsSDKInitialized(true);
          if (ICDeviceModule.getBleState) {
            const { state, enabled } = await ICDeviceModule.getBleState();
            console.log('Initial BLE state:', state, enabled);
            setBleState(state);
            setBleEnabled(enabled);
          }
        } catch (error) {
          console.error('Auto-initialization failed:', error);
          // Don't show alert here as user didn't explicitly request initialization
        }
      }
    };

    if (profile) autoInitialize(profile);

    // SDK initialization callback
    const sdkInitSub = emitter.addListener(
      'onSDKInit',
      ({ success }: { success: boolean }) => {
        console.log('SDK initialization result:', success);
        setIsSDKInitialized(success);
        initializationRef.current = success;

        if (!success) {
          Alert.alert(
            'SDK Initialization Failed',
            'Please check Bluetooth permissions and try again.',
          );
        }
      },
    );

    // Device found during scan
    const deviceFoundSub = emitter.addListener(
      'onDeviceFound',
      (device: ICDevice) => {
        console.log('Device found:', device.mac);
        setScannedDevices(prev => {
          const exists = prev.some(d => d.mac === device.mac);
          if (!exists) {
            return [...prev, device];
          }
          return prev;
        });
      },
    );

    // Device connection state changes
    const connectionSub = emitter.addListener(
      'onDeviceConnectionChanged',
      ({
        mac,
        state,
        isConnected,
      }: {
        mac: string;
        state: string;
        isConnected: boolean;
      }) => {
        console.log(
          `Device ${mac} connection changed to: ${state} (${isConnected ? 'connected' : 'disconnected'})`,
        );

        if (isConnected) {
          setConnectedDevices(prev => {
            if (!prev.some(d => d.mac === mac)) {
              const device = scannedDevices.find(d => d.mac === mac) || { mac };
              return [...prev, { ...device, isConnected: true }];
            }
            return prev.map(d =>
              d.mac === mac ? { ...d, isConnected: true } : d,
            );
          });
        } else {
          setConnectedDevices(prev => prev.filter(d => d.mac !== mac));
        }

        // Update scanned devices connection status
        setScannedDevices(prev =>
          prev.map(device =>
            device.mac === mac ? { ...device, isConnected } : device,
          ),
        );
      },
    );

    // Weight data callback
    const weightDataSub = emitter.addListener(
      'onReceiveWeightData',
      ({ device, data }: { device: ICDevice; data: ICWeightData }) => {
        console.log('Weight data received:', {
          device: device?.mac,
          weight: data?.weight,
          timestamp: new Date(data?.timestamp || Date.now()).toISOString(),
        });

        const measurement: ICWeightMeasurement = {
          device,
          data: {
            ...data,
            timestamp: data.timestamp || Date.now(),
          },
          timestamp: Date.now(),
        };

        setWeightData(prev => {
          const newData = [...prev, measurement];
          // Keep only last 100 measurements per device
          const deviceMeasurements = newData.filter(
            m => m.device.mac === device.mac,
          );
          const otherMeasurements = newData.filter(
            m => m.device.mac !== device.mac,
          );

          return [
            ...otherMeasurements,
            ...deviceMeasurements.slice(-50), // Keep last 50 per device
          ];
        });
      },
    );

    // Battery level callback
    const batterySub = emitter.addListener(
      'onReceiveBattery',
      ({ mac, battery }: { mac: string; battery: number }) => {
        console.log(`Device ${mac} battery level: ${battery}%`);
        setDeviceBatteryLevels(prev => ({
          ...prev,
          [mac]: battery,
        }));
      },
    );

    // Device info callback
    const deviceInfoSub = emitter.addListener(
      'onReceiveDeviceInfo',
      ({
        mac,
        deviceInfo: info,
      }: {
        mac: string;
        deviceInfo: ICDeviceInfo;
      }) => {
        console.log(`Device ${mac} info received:`, info);
        setDeviceInfo(prev => ({
          ...prev,
          [mac]: info,
        }));
      },
    );

    // Weight history data callback
    const weightHistorySub = emitter.addListener(
      'onReceiveWeightHistoryData',
      ({ mac, data }: { mac: string; data: ICWeightData }) => {
        console.log(`Device ${mac} history data:`, data);
        // Handle historical data if needed
      },
    );

    // RSSI callback
    const rssiSub = emitter.addListener(
      'onReceiveRSSI',
      ({ mac, rssi }: { mac: string; rssi: number }) => {
        console.log(`Device ${mac} RSSI: ${rssi}dBm`);
        // Update device RSSI in scanned devices
        setScannedDevices(prev =>
          prev.map(device =>
            device.mac === mac ? { ...device, rssi } : device,
          ),
        );
      },
    );

    // Device upgrade progress
    const upgradeProgressSub = emitter.addListener(
      'onReceiveUpgradePercent',
      ({
        mac,
        status,
        percent,
      }: {
        mac: string;
        status: string;
        percent: number;
      }) => {
        console.log(`Device ${mac} upgrade: ${status} ${percent}%`);
      },
    );

    // BLE state callback
    const bleStateSub = emitter.addListener(
      'onBleState',
      ({ state, enabled }: { state: string; enabled: boolean }) => {
        console.log('BLE state changed:', state, 'enabled:', enabled);
        setBleState(state);
        setBleEnabled(state === 'ICBleStatePoweredOn');
        if (!enabled) {
          // Alert.alert('Bluetooth Disabled', 'Please enable Bluetooth to use weight scales.');
          setIsScanning(false);
          setConnectedDevices([]);
        }
      },
    );

    return () => {
      console.log('Cleaning up ICDevice listeners...');

      // Clear scan timeout
      if (scanTimeoutRef.current) {
        clearTimeout(scanTimeoutRef.current);
      }

      // Remove all listeners
      sdkInitSub.remove();
      deviceFoundSub.remove();
      connectionSub.remove();
      weightDataSub.remove();
      batterySub.remove();
      deviceInfoSub.remove();
      weightHistorySub.remove();
      rssiSub.remove();
      upgradeProgressSub.remove();
      bleStateSub.remove();
    };
  }, [profile]);

  // Debug logging
  useEffect(() => {
    console.log('ICDevice Context State:', {
      scannedDevices: scannedDevices.length,
      connectedDevices: connectedDevices.length,
      weightData: weightData.length,
      isScanning,
      isSDKInitialized,
      bleState,
    });
  }, [
    scannedDevices,
    connectedDevices,
    weightData,
    isScanning,
    isSDKInitialized,
    bleState,
  ]);
  const setCurrentUserForAllDevices = async (profile: UserProfile) => {
    if (connectedDevices.length === 0) return;

    const gender = profile.gender === 'FEMALE' ? 'FEMALE' : 'MALE';
    const userInfo: ICUserInfo = {
      name: profile.username,
      age: calculateAge(profile.dob),
      height: profile.height,
      gender: gender,
    };
    console.log('user: ', userInfo);
    console.log('connected: ', connectedDevices);
    await Promise.all(
      connectedDevices.map(async device => {
        const connected = await isDeviceConnected(device.mac);
        console.log('device is connected: ', connected);
        setUserInfo(device.mac, userInfo).catch(err =>
          console.error(`Failed to set user for device ${device.mac}:`, err),
        );
        setCurrentUserInfo(device.mac, userInfo).catch(err =>
          console.error(
            `Failed to set current user for device ${device.mac}:`,
            err,
          ),
        );
      }),
    );

    console.log('Current user set for all connected devices');
  };
  // useEffect(() => {
  //   if (profile && connectedDevices.length > 0) {
  //     setCurrentUserForAllDevices(profile);
  //   }
  // }, [profile, connectedDevices]);

  return (
    <ICDeviceContext.Provider
      value={{
        scannedDevices,
        connectedDevices,
        weightData,
        isScanning,
        isSDKInitialized,
        bleState,
        deviceBatteryLevels,
        deviceInfo,
        bleEnabled,

        getLatestWeightForDevice,
        clearWeightData,

        initializeSDK,
        connectDevice,
        disconnectDevice,
        startScan,
        stopScan,
        clearScannedDevices,
        refreshDevices,
        isDeviceConnected,

        getBodyFatAlgorithmsManager,

        setUserInfo,
      }}>
      {children}
    </ICDeviceContext.Provider>
  );
}

export function useICDevice() {
  const context = useContext(ICDeviceContext);
  if (!context) {
    throw new Error('useICDevice must be used within an ICDeviceProvider');
  }
  return context;
}
