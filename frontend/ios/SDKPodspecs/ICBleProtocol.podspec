Pod::Spec.new do |s|
  s.name = 'ICBleProtocol'
  s.version = '1.0.0'
  s.summary = 'ICBleProtocol Framework'
  s.description = 'Vendor ICBleProtocol iOS SDK packaged as xcframework.'
  s.homepage = 'https://example.com'
  s.license = { :type => 'Proprietary' }
  s.author = { 'Vendor' => 'vendor@example.com' }
  s.platform = :ios, '11.0'
  s.vendored_frameworks = '../ICBleProtocol.xcframework'
  s.public_header_files = '../ICBleProtocol.xcframework/ios-arm64/Headers/**/*.h'
  s.requires_arc = true
  s.source = { :path => '../ICBleProtocol.xcframework' }
end
