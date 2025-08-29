package com.anonymous.frontend.device

import android.os.Build
import android.util.Log
import android.bluetooth.BluetoothAdapter
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.modules.core.DeviceEventManagerModule
import cn.icomon.icdevicemanager.ICDeviceManager
import cn.icomon.icdevicemanager.ICDeviceManagerDelegate
import cn.icomon.icdevicemanager.callback.ICScanDeviceDelegate
import cn.icomon.icdevicemanager.model.device.ICDevice
import cn.icomon.icdevicemanager.model.device.ICScanDeviceInfo
import cn.icomon.icdevicemanager.model.device.ICUserInfo
import cn.icomon.icdevicemanager.model.device.ICDeviceInfo
import cn.icomon.icdevicemanager.model.other.ICConstant
import cn.icomon.icdevicemanager.model.other.ICDeviceManagerConfig
import cn.icomon.icdevicemanager.model.data.ICWeightData
import cn.icomon.icdevicemanager.model.data.ICWeightCenterData
import cn.icomon.icdevicemanager.model.data.ICKitchenScaleData
import cn.icomon.icdevicemanager.model.data.ICCoordData
import cn.icomon.icdevicemanager.model.data.ICWeightHistoryData
import cn.icomon.icdevicemanager.model.data.ICRulerData
import cn.icomon.icdevicemanager.model.data.ICSkipData

data class Device(val mac: String, val name: String)

class ScanManager(private val reactContext: ReactApplicationContext) {

    companion object {
        const val TAG = "ICScanManager"
    }

    private val scannedDevices = mutableListOf<Device>()
    private val connectedDevices = mutableListOf<Device>()

    private val deviceManagerDelegate = object : ICDeviceManagerDelegate {
        override fun onInitFinish(success: Boolean) {
            Log.d(TAG, "SDK init finished: $success")
            emitToJS("onSDKInit", Arguments.createMap().apply { putBoolean("success", success) })
        }

        override fun onBleState(state: ICConstant.ICBleState) {
            Log.d(TAG, "BLE state: $state")
            emitToJS("onBleState", Arguments.createMap().apply { putString("state", state.name) })
        }

        override fun onDeviceConnectionChanged(device: ICDevice, state: ICConstant.ICDeviceConnectState) {
            val mac = device.macAddr
            val name = scannedDevices.find { it.mac == mac }?.name ?: "Unknown"
            Log.d(TAG, "Device ${device.macAddr} connection changed: $state")

            when (state) {
                ICConstant.ICDeviceConnectState.ICDeviceConnectStateConnected -> {
                    if (connectedDevices.none { it.mac == mac }) connectedDevices.add(Device(mac, name))
                }
                ICConstant.ICDeviceConnectState.ICDeviceConnectStateDisconnected -> {
                    connectedDevices.removeAll { it.mac == mac }
                }
            }

            emitToJS("onDeviceConnectionChanged", Arguments.createMap().apply {
                putString("mac", mac)
                putString("name", name)
                putString("state", state.name)
            })
        }

        override fun onReceiveUpgradePercent(device: ICDevice, status: ICConstant.ICUpgradeStatus, percent: Int) {}
        override fun onReceiveBattery(device: ICDevice, battery: Int, ext: Any?) {}
        override fun onReceiveDeviceInfo(device: ICDevice, deviceInfo: ICDeviceInfo) {}
        override fun onReceiveRSSI(device: ICDevice, rssi: Int) {}
        override fun onReceiveUserInfo(device: ICDevice, userInfo: ICUserInfo) {}
        override fun onReceiveUserInfoList(device: ICDevice, list: List<ICUserInfo>) {}
        override fun onReceiveHR(device: ICDevice, hr: Int) {}

        // Weight scale callbacks
        override fun onReceiveWeightData(device: ICDevice, data: ICWeightData) {}
        override fun onReceiveWeightCenterData(device: ICDevice, data: ICWeightCenterData) {}
        override fun onReceiveCoordData(device: ICDevice, data: ICCoordData) {}
        override fun onReceiveMeasureStepData(device: ICDevice, step: ICConstant.ICMeasureStep, data: Any?) {}
        override fun onReceiveWeightHistoryData(device: ICDevice, data: ICWeightHistoryData) {}
        override fun onReceiveWeightUnitChanged(device: ICDevice, unit: ICConstant.ICWeightUnit) {}

        // Kitchen scale callbacks
        override fun onReceiveKitchenScaleData(device: ICDevice, data: ICKitchenScaleData) {}
        override fun onReceiveKitchenScaleUnitChanged(device: ICDevice, unit: ICConstant.ICKitchenScaleUnit) {}

        // Ruler callbacks
        override fun onReceiveRulerData(device: ICDevice, data: ICRulerData) {}
        override fun onReceiveRulerHistoryData(device: ICDevice, data: ICRulerData) {}
        override fun onReceiveRulerUnitChanged(device: ICDevice, unit: ICConstant.ICRulerUnit) {}
        override fun onReceiveRulerMeasureModeChanged(device: ICDevice, mode: ICConstant.ICRulerMeasureMode) {}

        // Skipping rope callbacks
        override fun onReceiveSkipData(device: ICDevice, data: ICSkipData) {}
        override fun onReceiveHistorySkipData(device: ICDevice, data: ICSkipData) {}
        override fun onNodeConnectionChanged(device: ICDevice, nodeId: Int, state: ICConstant.ICDeviceConnectState) {}

        // Callbacks not in docs
        override fun onReceiveDebugData(device: ICDevice, type: Int, data: Any?) {}
        override fun onReceiveDeviceLightSetting(device: ICDevice, data: Any?) {}
        override fun onReceiveScanWifiInfo_W(device: ICDevice, ssid: String, method: Int, rssi: Int) {}
        override fun onReceiveCurrentWifiInfo_W(device: ICDevice, status: Int, ip: String, ssid: String, rssi: Int) {}
        override fun onReceiveBindState_W(device: ICDevice, status: Int) {}
        override fun onReceiveConfigWifiResult(device: ICDevice, state: ICConstant.ICConfigWifiState) {}
    }

    private val scanDelegate = object : ICScanDeviceDelegate {
        override fun onScanResult(deviceInfo: ICScanDeviceInfo) {
            val mac = deviceInfo.macAddr
            val name = deviceInfo.name ?: "Unknown"

            if (scannedDevices.none { it.mac == mac }) scannedDevices.add(Device(mac, name))

            emitToJS("onScannedDevices", Arguments.createArray().apply {
                pushMap(Arguments.createMap().apply {
                    putString("mac", mac)
                    putString("name", name)
                    putInt("bindStatus", deviceInfo.bindStatus)
                })
            })
        }
    }

    private fun emitToJS(event: String, params: Any?) {
        reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(event, params)
    }

    private val bluetoothReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val state = intent?.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
            when (state) {
                BluetoothAdapter.STATE_ON -> emitToJS(
                    "onBleState",
                    Arguments.createMap().apply { putString("state", "ICBleStatePoweredOn") }
                )
                BluetoothAdapter.STATE_OFF -> emitToJS(
                    "onBleState",
                    Arguments.createMap().apply { putString("state", "ICBleStatePoweredOff") }
                )
            }
        }
    }

    fun initSDK() {
        val config = ICDeviceManagerConfig()
        config.context = reactContext.applicationContext
        ICDeviceManager.shared().setDelegate(deviceManagerDelegate)
        ICDeviceManager.shared().initMgrWithConfig(config)
        Log.d(TAG, "SDK initialized")

        val filter = IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED)
        reactContext.registerReceiver(bluetoothReceiver, filter)
    }

    fun cleanup() {
        reactContext.unregisterReceiver(bluetoothReceiver)
    }

    fun startScan() {
        ICDeviceManager.shared().scanDevice(scanDelegate)
        Log.d(TAG, "Scan started")
    }

    fun stopScan() {
        ICDeviceManager.shared().stopScan()
        Log.d(TAG, "Scan stopped")
    }

    fun addDevice(mac: String, deviceName: String) {
        val device = ICDevice().apply { setMacAddr(mac) }
        val name = scannedDevices.find { it.mac == mac }?.name ?: deviceName
        ICDeviceManager.shared().addDevice(device) { _, _ ->
            emitToJS("onDeviceAdded", Arguments.createMap().apply {
                putString("mac", mac)
                putString("name", name)
            })
        }
    }

    fun removeDevice(mac: String) {
        val device = ICDevice().apply { setMacAddr(mac) }
        ICDeviceManager.shared().removeDevice(device) { _, _ ->
            emitToJS("onDeviceRemoved", Arguments.createMap().apply { putString("mac", mac) })
        }
    }

    fun getScannedDevices(): List<Device> = scannedDevices
    fun getConnectedDevices(): List<Device> = connectedDevices
}
