import React, { createContext, useContext, useEffect, useState } from "react";
import { NativeModules, NativeEventEmitter } from "react-native";
import { Device } from '@/types/icdevice-types';

const { ICDeviceModule } = NativeModules;
const emitter = ICDeviceModule ? new NativeEventEmitter(ICDeviceModule) : null;

interface ICDeviceContextType {
  scannedDevices: Device[];
  addedDevices: Device[];
  connectedDevices: Device[];
  bluetoothEnabled: boolean;
  startScan: () => void;
  stopScan: () => void;
  addDevice: (device: Device) => void;
  removeDevice: (mac: string) => void;
}

export const ICDeviceContext = createContext<ICDeviceContextType>({
  scannedDevices: [],
  addedDevices: [],
  connectedDevices: [],
  bluetoothEnabled: false,
  startScan: () => {},
  stopScan: () => {},
  addDevice: () => {},
  removeDevice: () => {},
});

export function ICDeviceProvider({ children }: { children: React.ReactNode }) {
  const [scannedDevices, setScannedDevices] = useState<Device[]>([]);
  const [addedDevices, setAddedDevices] = useState<Device[]>([]);
  const [connectedDevices, setConnectedDevices] = useState<Device[]>([]);
  const [bluetoothEnabled, setBluetoothEnabled] = useState(false);

  useEffect(() => {
    if (!ICDeviceModule || !emitter) return;

    ICDeviceModule.initializeSDK();

    const sdkInitSub = emitter.addListener("onSDKInit", ({ success }) => {
      if (success) {
        console.log("initialised")
        ICDeviceModule.getConnectedDevices();
      }
    });

    const bleStateSub = emitter.addListener("onBleState", ({ state }) => {
      setBluetoothEnabled(state === "ICBleStatePoweredOn");
    })

    const scannedSub = emitter.addListener("onScannedDevices", setScannedDevices);

    const connectedSub = emitter.addListener("onConnectedDevices", setConnectedDevices);

    const connectionSub = emitter.addListener("onDeviceConnectionChanged", ({ mac, name, state }: {mac: string, name: string, state: string }) => {
      setConnectedDevices(prev => {
        if (state === "ICDeviceConnectStateConnected") {
          if (!prev.some(d => d.mac === mac)) return [...prev, { mac, name }];
        } else if (state === "ICDeviceConnectStateDisconnected") {
          return prev.filter(d => d.mac !== mac);
        }
        return prev;
      });
    });

    const addedSub = emitter.addListener("onDeviceAdded", ({ mac }) => {
      setAddedDevices(prev => {
        if (!prev.some(d => d.mac === mac)) {
          const device = scannedDevices.find(d => d.mac === mac) || { mac };
          return [...prev, device];
        }
        return prev;
      });
    });

    const removedSub = emitter.addListener("onDeviceRemoved", ({ mac }) => {
      setAddedDevices(prev => prev.filter(d => d.mac !== mac));
      setConnectedDevices(prev => prev.filter(d => d.mac !== mac));
    });

    return () => {
      sdkInitSub.remove();
      bleStateSub.remove()
      scannedSub.remove();
      connectedSub.remove();
      connectionSub.remove();
      addedSub.remove();
      removedSub.remove();
    };
  }, []);

  const startScan = () => ICDeviceModule.startScan();
  const stopScan = () => ICDeviceModule.stopScan();
  const addDevice = (device: Device) => ICDeviceModule.addDevice(device.mac, device.name);
  const removeDevice = (mac: string) => ICDeviceModule.removeDevice(mac);

  return (
    <ICDeviceContext.Provider
      value={{
        scannedDevices,
        addedDevices,
        connectedDevices,
        bluetoothEnabled,
        startScan,
        stopScan,
        addDevice,
        removeDevice,
      }}
    >
      {children}
    </ICDeviceContext.Provider>
  );
}

export function useICDevice() {
  return useContext(ICDeviceContext);
}
