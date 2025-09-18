Pod::Spec.new do |s|
  s.name = 'ICLogger'
  s.version = '1.0.0'
  s.summary = 'ICLogger Framework'
  s.description = 'Vendor ICLogger iOS SDK packaged as xcframework.'
  s.homepage = 'https://example.com'
  s.license = { :type => 'Proprietary' }
  s.author = { 'Vendor' => 'vendor@example.com' }
  s.platform = :ios, '11.0'
  s.vendored_frameworks = '../ICLogger.xcframework'
  s.public_header_files = '../ICLogger.xcframework/ios-arm64/Headers/**/*.h'
  s.requires_arc = true
  s.source = { :path => '../ICLogger.xcframework' }
end
