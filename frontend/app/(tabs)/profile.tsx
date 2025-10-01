import { useEffect, useState } from 'react';
import { Image, Pressable, ActivityIndicator, Alert, Animated } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useAuth } from '@/context/AuthContext';
import { supabase } from '@/lib/supabase';
import { ThemedSafeAreaView, View, Text } from '@/components/Themed';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Buffer } from 'buffer';
import uuid from 'react-native-uuid';
import { useThemeMode } from '@/context/ThemeContext';
import ThemeToggle from '@/components/ThemeToggle';
import { LinearGradient } from 'expo-linear-gradient';

export default function ProfileScreen() {
  const { mode, setMode } = useThemeMode();
  const { profile, user, profileLoading, loading, refreshProfile } = useAuth();
  const [photo, setPhoto] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const router = useRouter();
  
  // Animation values
  const fadeAnim = useState(new Animated.Value(0))[0];
  const slideAnim = useState(new Animated.Value(50))[0];
  const scaleAnim = useState(new Animated.Value(0.95))[0];

  const email = user?.email ?? undefined;
  const memberSinceIso = profile?.created_at ?? user?.created_at;
  const memberSince = memberSinceIso
    ? new Date(memberSinceIso).toLocaleDateString()
    : undefined;
  const isBusy = loading || profileLoading;

  useEffect(() => {
    if (!loading && !user) router.replace('/login');
  }, [user, loading]);

  useEffect(() => {
    const loadPhoto = async () => {
      const storedBase64 = await AsyncStorage.getItem('profile_photo');
      if (storedBase64) {
        setPhoto(storedBase64);
      } else if (profile?.avatar_url) {
        const { data: urlData } = supabase.storage
          .from('avatars')
          .getPublicUrl(profile.avatar_url);
        if (urlData?.publicUrl) setPhoto(urlData.publicUrl);
      }
    };
    loadPhoto();
  }, [profile]);

  // Start animations when component mounts
  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  // Pick image from gallery
  const pickImage = async () => {
    try {
      const { status } =
        await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission required',
          'Please allow access to your photo library.',
        );
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
        base64: true,
      });

      if (
        result.assets &&
        result.assets.length > 0 &&
        result.assets[0].base64
      ) {
        const base64Image = `data:image/png;base64,${result.assets[0].base64}`;
        setPhoto(base64Image); // show immediately
        await AsyncStorage.setItem('profile_photo', base64Image); // persist locally
        await uploadPhoto(result.assets[0].base64); // upload to Supabase
      }
    } catch (err) {
      console.log('Error picking image:', err);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  const uploadPhoto = async (base64Data: string) => {
    if (!user) return;
    try {
      setUploading(true);
      const id = uuid.v4();
      const fileName = `avatars/${user.id}-${id}.png`;
      const buffer = Buffer.from(base64Data, 'base64');

      const { error: uploadError } = await supabase.storage
        .from('avatars')
        .upload(fileName, buffer, { contentType: 'image/png', upsert: true });

      if (uploadError) throw uploadError;

      // Get public URL and update local state
      const { data: urlData } = supabase.storage
        .from('avatars')
        .getPublicUrl(fileName);
      if (urlData?.publicUrl) setPhoto(urlData.publicUrl);

      // Update profile table
      const { error: dbError } = await supabase
        .from('user_profiles')
        .update({ avatar_url: fileName })
        .eq('user_id', user.id);

      if (dbError) throw dbError;

      if (refreshProfile) await refreshProfile();
      console.log('Photo uploaded successfully');
    } catch (err) {
      console.log('Error uploading photo:', err);
      Alert.alert('Error', 'Failed to upload photo. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  // Logout
  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      router.replace('/login');
    } catch (err) {
      console.log('Logout error:', err);
      Alert.alert('Error', 'Failed to log out.');
    }
  };

  const handleEditProfile = () => {
    // Navigate to edit profile screen - you can implement this
    Alert.alert('Edit Profile', 'Edit profile feature coming soon!');
  };

  const handleSettings = () => {
    // Navigate to settings screen - you can implement this
    Alert.alert('Settings', 'Settings feature coming soon!');
  };

  if (!user) {
    return (
      <ThemedSafeAreaView>
        <View className="flex-1 items-center justify-center">
          <ActivityIndicator size="large" />
        </View>
      </ThemedSafeAreaView>
    );
  }

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1">
      {/* Elegant Wave-like Gradient Background */}
      <Animated.View 
        style={{ 
          opacity: fadeAnim,
          transform: [{ scale: scaleAnim }]
        }}
        className="absolute top-0 left-0 right-0 h-48 overflow-hidden"
      >
        <LinearGradient
          colors={mode === 'dark' 
            ? ['#3b82f6', '#60a5fa', '#93c5fd'] 
            : ['#3b82f6', '#60a5fa', '#93c5fd']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          className="h-full"
        />
        
        {/* Wave Pattern Overlay */}
        <View className="absolute -bottom-10 left-0 right-0 h-20">
          <View 
  className="absolute -top-10 left-0 right-0 h-20 bg-background rounded-t-[40px]"
  style={{
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  }}
/>
        </View>
      </Animated.View>

      <Animated.ScrollView 
        style={{ 
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }]
        }}
        className="flex-1 px-4"
        showsVerticalScrollIndicator={false}
      >
        {/* Profile Header */}
        <View className="items-center mt-12 mb-8">
          <Pressable
            onPress={pickImage}
            disabled={uploading}
            className="overflow-hidden rounded-full border-4 border-background-0 shadow-2xl"
            style={{ 
              width: 120, 
              height: 120,
              shadowColor: '#000',
              shadowOffset: { width: 0, height: 8 },
              shadowOpacity: 0.2,
              shadowRadius: 16,
              elevation: 8,
            }}
          >
            <Image
              source={{
                uri: photo
                  ? photo
                  : `https://ui-avatars.com/api/?name=${encodeURIComponent(profile?.username || email || 'User')}&background=random&color=fff&bold=true&size=256`,
              }}
              style={{ width: 112, height: 112, borderRadius: 56 }}
            />
            <View
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: 112,
                height: 112,
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: 'rgba(0,0,0,0.4)',
                borderRadius: 56,
              }}>
              {uploading ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Ionicons name="camera-outline" size={28} color="#fff" />
              )}
            </View>
          </Pressable>

          <Text className="mt-4 font-bodyBold text-head1 text-primary-500">
            {profile?.username || 'Guest User'}
          </Text>
          <Text className="text-base text-primary-300">
            {isBusy ? 'Loading...' : (email ?? 'No email')}
          </Text>
        </View>

        {/* Profile Stats Grid */}
        <View className="flex-row justify-between mb-6">
          <View className="flex-1 mr-2 rounded-2xl bg-background-0 p-4 shadow-md items-center">
            <Ionicons name="calendar-outline" size={24} color="#3b82f6" />
            <Text className="font-bodyBold text-body2 text-primary-500 mt-1">
              Member Since
            </Text>
            <Text className="font-body text-sm text-primary-400 text-center">
              {isBusy ? 'Loading...' : (memberSince ?? 'N/A')}
            </Text>
          </View>
          
          <View className="flex-1 ml-2 rounded-2xl bg-background-0 p-4 shadow-md items-center">
            <Ionicons name="time-outline" size={24} color="#10b981" />
            <Text className="font-bodyBold text-body2 text-primary-500 mt-1">
              Status
            </Text>
            <Text className="font-body text-sm text-green-600 text-center">
              Active
            </Text>
          </View>
        </View>

        {/* Profile Details Card */}
        <Animated.View 
          style={{ transform: [{ scale: scaleAnim }] }}
          className="mb-6 rounded-3xl bg-background-0 p-6 shadow-lg"
        >
          <Text className="font-bodySemiBold text-body1 text-primary-500 mb-4">
            Personal Information
          </Text>
          
          <View className="space-y-4">
            <View className="flex-row justify-between items-center py-2">
              <View className="flex-row items-center">
                <Ionicons name="person-outline" size={20} color="#6b7280" />
                <Text className="font-body text-primary-400 ml-2">Username</Text>
              </View>
              <Text className="font-bodyBold text-primary-500">
                {profile?.username || 'Not set'}
              </Text>
            </View>

            <View className="h-[1px] bg-primary-100" />

            <View className="flex-row justify-between items-center py-2">
              <View className="flex-row items-center">
                <Ionicons name="calendar-outline" size={20} color="#6b7280" />
                <Text className="font-body text-primary-400 ml-2">Date of Birth</Text>
              </View>
              <Text className="font-bodyBold text-primary-500">
                {profile?.dob || 'Not set'}
              </Text>
            </View>

            <View className="h-[1px] bg-primary-100" />

            <View className="flex-row justify-between items-center py-2">
              <View className="flex-row items-center">
                <Ionicons name="male-female-outline" size={20} color="#6b7280" />
                <Text className="font-body text-primary-400 ml-2">Gender</Text>
              </View>
              <Text className="font-bodyBold text-primary-500">
                {profile?.gender || 'Not set'}
              </Text>
            </View>
          </View>
        </Animated.View>

        {/* Action Buttons */}
        <View className="mb-6 rounded-3xl bg-background-0 p-4 shadow-lg">
          <Pressable 
            onPress={handleEditProfile}
            className="flex-row items-center justify-between py-4 px-2 active:opacity-70"
          >
            <View className="flex-row items-center">
              <Ionicons name="create-outline" size={22} color="#6b7280" />
              <Text className="font-bodySemiBold text-body2 text-primary-500 ml-3">
                Edit Profile
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#6b7280" />
          </Pressable>

          <View className="h-[1px] bg-primary-100" />

          <Pressable 
            onPress={handleSettings}
            className="flex-row items-center justify-between py-4 px-2 active:opacity-70"
          >
            <View className="flex-row items-center">
              <Ionicons name="settings-outline" size={22} color="#6b7280" />
              <Text className="font-bodySemiBold text-body2 text-primary-500 ml-3">
                Settings
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#6b7280" />
          </Pressable>
        </View>

        {/* Theme Toggle - Simple version like before */}
        <View className="mb-8 gap-1">
          <Text className="font-bodySemiBold text-body2 text-primary-300">
            Theme
          </Text>
          <ThemeToggle />
        </View>
      </Animated.ScrollView>

      {/* Logout Button */}
      <Animated.View 
        style={{ opacity: fadeAnim }}
        className="px-4 pb-8 pt-4"
      >
        <Pressable 
          onPress={handleLogout}
          className="bg-red-500 rounded-2xl py-4 shadow-lg active:opacity-80 active:scale-95"
        >
          <View className="flex-row items-center justify-center">
            <Ionicons name="log-out-outline" size={20} color="#fff" />
            <Text className="text-center font-bodyBold text-body2 text-background-0 ml-2">
              Log out
            </Text>
          </View>
        </Pressable>
      </Animated.View>
    </ThemedSafeAreaView>
  );
}