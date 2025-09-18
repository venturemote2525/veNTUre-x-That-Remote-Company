Pod::Spec.new do |s|
  s.name = 'ICBodyFatAlgorithms'
  s.version = '1.0.0'
  s.summary = 'ICBodyFatAlgorithms Framework'
  s.description = 'Vendor ICBodyFatAlgorithms iOS SDK packaged as xcframework.'
  s.homepage = 'https://example.com'
  s.license = { :type => 'Proprietary' }
  s.author = { 'Vendor' => 'vendor@example.com' }
  s.platform = :ios, '11.0'
  s.vendored_frameworks = '../ICBodyFatAlgorithms.xcframework'
  s.public_header_files = '../ICBodyFatAlgorithms.xcframework/ios-arm64/Headers/**/*.h'
  s.requires_arc = true
  s.source = { :path => '../ICBodyFatAlgorithms.xcframework' }
end
