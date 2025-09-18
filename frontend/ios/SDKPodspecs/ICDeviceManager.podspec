Pod::Spec.new do |s|
  s.name = 'ICDeviceManager'
  s.version = '1.0.0'
  s.summary = 'ICDeviceManager Framework'
  s.description = 'Vendor ICDeviceManager iOS SDK packaged as xcframework.'
  s.homepage = 'https://example.com'
  s.license = { :type => 'Proprietary' }
  s.author = { 'Vendor' => 'vendor@example.com' }
  s.platform = :ios, '11.0'
  s.vendored_frameworks = '../ICDeviceManager.xcframework'
  s.public_header_files = '../ICDeviceManager.xcframework/ios-arm64/Headers/**/*.h'
  s.requires_arc = true
  s.source = { :path => '../ICDeviceManager.xcframework' }
end
