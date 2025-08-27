package com.anonymous.frontend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
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

class ICDeviceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val TAG = "ICDeviceModule"
        const val CODE_REQUEST_PERMISSION = 1001
    }

    private val scannedDevices = mutableListOf<ICDevice>()
    private val connectedDevices = mutableListOf<ICDevice>()

    private lateinit var deviceManagerDelegate: ICDeviceManagerDelegate
    private lateinit var scanDelegate: ICScanDeviceDelegate

    init {
        setupDelegates()
    }

    override fun getName(): String = "ICDeviceModule"

    private fun emitToJS(event: String, params: Any?) {
        reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(event, params)
    }

    private fun checkBLEPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.checkSelfPermission(
                reactContext.currentActivity!!,
                Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(
                        reactContext.currentActivity!!,
                        Manifest.permission.BLUETOOTH_SCAN
                    ) == PackageManager.PERMISSION_GRANTED
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.checkSelfPermission(
                reactContext.currentActivity!!,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(
                        reactContext.currentActivity!!,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    ) == PackageManager.PERMISSION_GRANTED
        } else true
    }

    private fun requestBLEPermission() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN)
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        ActivityCompat.requestPermissions(
            reactContext.currentActivity!!,
            permissions,
            CODE_REQUEST_PERMISSION
        )
    }

    private fun setupDelegates() {
        // Device manager callbacks
        deviceManagerDelegate = object : ICDeviceManagerDelegate {
            override fun onInitFinish(success: Boolean) {
                Log.d(TAG, "SDK init finished: $success")
                emitToJS("onSDKInit", Arguments.createMap().apply { putBoolean("success", success) })
            }

            override fun onBleState(state: ICConstant.ICBleState) {
                Log.d(TAG, "BLE state: $state")
                emitToJS("onBleState", Arguments.createMap().apply { putString("state", state.name) })
            }

            override fun onDeviceConnectionChanged(device: ICDevice, state: ICConstant.ICDeviceConnectState) {
                Log.d(TAG, "Device ${device.macAddr} connection changed: $state")

                when (state) {
                    ICConstant.ICDeviceConnectState.ICDeviceConnectStateConnected -> {
                        if (connectedDevices.none { it.macAddr == device.macAddr }) {
                            connectedDevices.add(device)
                        }
                    }
                    ICConstant.ICDeviceConnectState.ICDeviceConnectStateDisconnected -> {
                        connectedDevices.removeAll { it.macAddr == device.macAddr }
                    }
                }

                emitToJS("onDeviceConnectionChanged", Arguments.createMap().apply {
                    putString("mac", device.macAddr)
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

        // Scan device callback
        scanDelegate = object : ICScanDeviceDelegate {
            override fun onScanResult(deviceInfo: ICScanDeviceInfo) {
                val device = ICDevice()
                device.setMacAddr(deviceInfo.macAddr)

                // Add device if not in list
                if (scannedDevices.none { it.macAddr == device.macAddr }) {
                    scannedDevices.add(device)
                }
                ICDeviceManager.shared().addDevice(device, object : ICConstant.ICAddDeviceCallBack {
                    override fun onCallBack(device: ICDevice, code: ICConstant.ICAddDeviceCallBackCode) {
                        Log.d(TAG, "Device added: ${device.macAddr} -> $code")
                        emitToJS("onDeviceAdded", Arguments.createMap().apply {
                            putString("mac", device.macAddr)
                            putString("status", code.toString())
                        })
                    }
                })
            }
        }
    }

    private fun initSDK() {
        if (!checkBLEPermission()) {
            requestBLEPermission()
            return
        }

        val config = ICDeviceManagerConfig()
        config.context = reactContext.applicationContext
        ICDeviceManager.shared().setDelegate(deviceManagerDelegate)
        ICDeviceManager.shared().initMgrWithConfig(config)
        Log.d(TAG, "SDK initialized")
    }

    @ReactMethod
    fun initializeSDK() {
        val activity = reactContext.currentActivity
        if (activity == null) {
            Log.w(TAG, "Activity is null, cannot init SDK yet")
            return
        }

        if (!checkBLEPermission()) {
            requestBLEPermission()
            return
        }

        val config = ICDeviceManagerConfig()
        config.context = reactContext.applicationContext
        ICDeviceManager.shared().setDelegate(deviceManagerDelegate)
        ICDeviceManager.shared().initMgrWithConfig(config)
        Log.d(TAG, "SDK initialized")
    }

    @ReactMethod
    fun startScan() {
        if (!checkBLEPermission()) {
            requestBLEPermission()
            return
        }
        ICDeviceManager.shared().scanDevice(scanDelegate)
        Log.d(TAG, "Scan started")
    }

    @ReactMethod
    fun stopScan() {
        ICDeviceManager.shared().stopScan()
        Log.d(TAG, "Scan stopped")
    }

    @ReactMethod
    fun getScannedDevices() = emitToJS(
        "onScannedDevices",
        Arguments.createArray().apply {
            scannedDevices.forEach { device ->
                pushMap(Arguments.createMap().apply {
                    putString("mac", device.macAddr)
                })
            }
        }
    )

    @ReactMethod
    fun getConnectedDevices() = emitToJS(
        "onConnectedDevices",
        Arguments.createArray().apply {
            connectedDevices.forEach { device ->
                pushMap(Arguments.createMap().apply {
                    putString("mac", device.macAddr)
                })
            }
        }
    )
}
