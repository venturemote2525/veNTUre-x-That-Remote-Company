//
//  ICDeviceModule.swift
//  frontend
//

import ICDeviceManager
import ICBodyFatAlgorithms
import ICBleProtocol
import ICLogger

import Foundation
import React
import ICDeviceManager

@objc(ICDeviceModule)
class ICDeviceModule: RCTEventEmitter {
  
  private var hasListeners = false
  private var connectedDevices: [String: ICDevice] = [:]
  private var scanDelegate: ScanDeviceDelegate?
  private var deviceDelegate: DeviceManagerDelegate?
  
  override init() {
    super.init()
  }
  
  // MARK: - RCTEventEmitter
  
  override func supportedEvents() -> [String]! {
    return [
      "onDeviceFound",
      "onDeviceConnected",
      "onDeviceDisconnected",
      "onWeightDataReceived",
      "onRulerDataReceived",
      "onKitchenDataReceived",
      "onCoordDataReceived",
      "onScanFinished",
      "onInitialized"
    ]
  }
  
  override func startObserving() {
    hasListeners = true
  }
  
  override func stopObserving() {
    hasListeners = false
  }
  
  override static func requiresMainQueueSetup() -> Bool {
    return true
  }
  
  // MARK: - Exposed Methods
  
  @objc
  func initializeSDK(_ resolve: @escaping RCTPromiseResolveBlock,
                     rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      
      // Initialize delegates
      self.deviceDelegate = DeviceManagerDelegate(module: self)
      self.scanDelegate = ScanDeviceDelegate(module: self)
      
      // Set delegate and initialize
      ICDeviceManager.shared().delegate = self.deviceDelegate
      ICDeviceManager.shared().initMgr()
      
      print("✅ ICDeviceManager initialized")
      
      if self.hasListeners {
        self.sendEvent(withName: "onInitialized", body: ["success": true])
      }
      
      resolve(["success": true])
    }
  }
  
  @objc
  func startScan(_ resolve: @escaping RCTPromiseResolveBlock,
                 rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      
      ICDeviceManager.shared().scanDevice(self.scanDelegate)
      print("🔍 Started scanning for devices")
      resolve(["scanning": true])
    }
  }
  
  @objc
  func stopScan(_ resolve: @escaping RCTPromiseResolveBlock,
                rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async {
      ICDeviceManager.shared().stopScan()
      print("🛑 Stopped scanning for devices")
      resolve(["scanning": false])
    }
  }
  
  @objc
  func addDevice(_ macAddr: String,
                 resolver resolve: @escaping RCTPromiseResolveBlock,
                 rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self else { return }
      
      let device = ICDevice()
      device.macAddr = macAddr
      
      ICDeviceManager.shared().add(device) { (device, code) in
        if code == .success {
          self.connectedDevices[macAddr] = device
          print("✅ Device added successfully: \(macAddr)")
          resolve(["success": true, "macAddr": macAddr])
        } else {
          print("❌ Failed to add device: \(macAddr), code: \(code.rawValue)")
          reject("ADD_DEVICE_ERROR", "Failed to add device", nil)
        }
      }
    }
  }
  
  @objc
  func removeDevice(_ macAddr: String,
                    resolver resolve: @escaping RCTPromiseResolveBlock,
                    rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self,
            let device = self.connectedDevices[macAddr] else {
        reject("DEVICE_NOT_FOUND", "Device not found", nil)
        return
      }
      
      ICDeviceManager.shared().remove(device) { (device, code) in
        if code == .success {
          self.connectedDevices.removeValue(forKey: macAddr)
          print("✅ Device removed successfully: \(macAddr)")
          resolve(["success": true, "macAddr": macAddr])
        } else {
          print("❌ Failed to remove device: \(macAddr)")
          reject("REMOVE_DEVICE_ERROR", "Failed to remove device", nil)
        }
      }
    }
  }
  
  @objc
  func updateUserInfo(_ userInfo: NSDictionary,
                      resolver resolve: @escaping RCTPromiseResolveBlock,
                      rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async {
      let user = ICUserInfo()
      
      // Parse user info from dictionary
      if let height = userInfo["height"] as? NSNumber {
        user.height = height.intValue
      }
      if let weight = userInfo["weight"] as? Double {
        user.weight = weight
      }
      if let age = userInfo["age"] as? NSNumber {
        user.age = age.intValue
      }
      if let sex = userInfo["sex"] as? NSNumber {
        user.sex = ICUserSex(rawValue: sex.intValue) ?? .male
      }
      if let weightUnit = userInfo["weightUnit"] as? NSNumber {
        user.weightUnit = ICWeightUnit(rawValue: weightUnit.intValue) ?? .kg
      }
      if let rulerUnit = userInfo["rulerUnit"] as? NSNumber {
        user.rulerUnit = ICRulerUnit(rawValue: rulerUnit.intValue) ?? .cm
      }
      if let kitchenUnit = userInfo["kitchenUnit"] as? NSNumber {
        user.kitchenUnit = ICKitchenScaleUnit(rawValue: kitchenUnit.intValue) ?? .g
      }
      if let peopleType = userInfo["peopleType"] as? NSNumber {
        user.peopleType = ICPeopleType(rawValue: peopleType.intValue) ?? .normal
      }
      
      ICDeviceManager.shared().updateUserInfo(user)
      print("✅ User info updated")
      resolve(["success": true])
    }
  }
  
  @objc
  func setScaleUnit(_ macAddr: String,
                    unit: NSInteger,
                    resolver resolve: @escaping RCTPromiseResolveBlock,
                    rejecter reject: @escaping RCTPromiseRejectBlock) {
    DispatchQueue.main.async { [weak self] in
      guard let self = self,
            let device = self.connectedDevices[macAddr] else {
        reject("DEVICE_NOT_FOUND", "Device not found", nil)
        return
      }
      
      let weightUnit = ICWeightUnit(rawValue: unit) ?? .kg
      
      ICDeviceManager.shared().getSettingManager().setScaleUnit(device, unit: weightUnit) { code in
        if code == .success {
          print("✅ Scale unit set successfully")
          resolve(["success": true])
        } else {
          print("❌ Failed to set scale unit")
          reject("SET_UNIT_ERROR", "Failed to set scale unit", nil)
        }
      }
    }
  }
  
  // MARK: - Helper Methods
  
  func emitEvent(name: String, body: Any?) {
    if hasListeners {
      sendEvent(withName: name, body: body)
    }
  }
}

// MARK: - Delegate Classes

class DeviceManagerDelegate: NSObject, ICDeviceManagerDelegate {
  weak var module: ICDeviceModule?
  
  init(module: ICDeviceModule) {
    self.module = module
    super.init()
  }
  
  func onInitFinish() {
    print("✅ ICDeviceManager init finished")
    module?.emitEvent(name: "onInitialized", body: ["success": true])
  }
  
  func onBleState(_ state: ICBleState) {
    print("📶 Bluetooth state: \(state.rawValue)")
  }
  
  func onDeviceConnectionChanged(_ device: ICDevice, state: ICDeviceConnectState) {
    let connected = (state == .connected)
    let body: [String: Any] = [
      "macAddr": device.macAddr ?? "",
      "connected": connected,
      "state": state.rawValue
    ]
    
    if connected {
      module?.emitEvent(name: "onDeviceConnected", body: body)
    } else {
      module?.emitEvent(name: "onDeviceDisconnected", body: body)
    }
  }
  
  func onReceiveWeightData(_ device: ICDevice, data: ICWeightData) {
    let body: [String: Any] = [
      "macAddr": device.macAddr ?? "",
      "weight_kg": data.weight_kg,
      "weight_lb": data.weight_lb,
      "weight_st": data.weight_st,
      "weight_jin": data.weight_jin,
      "isStabilized": data.isStabilized,
      "time": data.time,
      "imp": data.imp
    ]
    
    module?.emitEvent(name: "onWeightDataReceived", body: body)
  }
  
  func onReceiveKitchenScaleData(_ device: ICDevice, data: ICKitchenScaleData) {
    let body: [String: Any] = [
      "macAddr": device.macAddr ?? "",
      "value_g": data.