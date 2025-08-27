import { createContext, useContext, useEffect, useState } from "react";
import { NativeModules, NativeEventEmitter } from "react-native";

const { ICDeviceModule } = NativeModules;
const emitter = ICDeviceModule ? new NativeEventEmitter(ICDeviceModule) : null;

interface ICDeviceContextType {
  scannedDevices: any[];
  connectedDevices: any[];
}

const ICDeviceContext = createContext<ICDeviceContextType>({
  scannedDevices: [],
  connectedDevices: [],
});

export function ICDeviceProvider({ children }: { children: React.ReactNode }) {
  const [scannedDevices, setScannedDevices] = useState<any[]>([]);
  const [connectedDevices, setConnectedDevices] = useState<any[]>([]);

  useEffect(() => {
    if (!ICDeviceModule || !emitter) return;

    ICDeviceModule.initializeSDK();
    ICDeviceModule.startScan();

    // Initial fetch
    ICDeviceModule.getConnectedDevices();
    ICDeviceModule.getScannedDevices();

    const scannedSub = emitter.addListener("onScannedDevices", (data) => {
      setScannedDevices(data);
    });

    const connectionSub = emitter.addListener(
      "onDeviceConnectionChanged",
      ({ mac, state }: { mac: string; state: string }) => {
        setConnectedDevices((prev) => {
          if (state === "ICDeviceConnectStateConnected") {
            if (!prev.some((d) => d.mac === mac)) {
              return [...prev, { mac }];
            }
          } else if (state === "ICDeviceConnectStateDisconnected") {
            return prev.filter((d) => d.mac !== mac);
          }
          return prev;
        });
      }
    );

    return () => {
      scannedSub.remove();
      connectionSub.remove();
    };
  }, []);

  return (
    <ICDeviceContext.Provider value={{ scannedDevices, connectedDevices }}>
      {children}
    </ICDeviceContext.Provider>
  );
}

export function useICDevice() {
  return useContext(ICDeviceContext);
}
