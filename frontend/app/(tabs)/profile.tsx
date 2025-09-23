import { useEffect, useState } from 'react';
import { Image, Pressable, ActivityIndicator, Alert } from 'react-native';
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

export default function ProfileScreen() {
  const { mode, setMode } = useThemeMode();
  const { profile, user, profileLoading, loading, refreshProfile } = useAuth();
  const [photo, setPhoto] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const router = useRouter();

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
    <ThemedSafeAreaView edges={['top']} className="justify-between p-4">
      <View className="flex-1">
        <View className="mb-6 items-center">
          <Pressable
            onPress={pickImage}
            disabled={uploading}
            className="overflow-hidden rounded-full"
            style={{ width: 96, height: 96 }}>
            <Image
              source={{
                uri: photo
                  ? photo
                  : `https://ui-avatars.com/api/?name=${encodeURIComponent(profile?.username || email || 'User')}&background=random`,
              }}
              style={{ width: 96, height: 96, borderRadius: 48 }}
            />
            <View
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: 96,
                height: 96,
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: 48,
              }}>
              {uploading ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Ionicons name="camera-outline" size={24} color="#fff" />
              )}
            </View>
          </Pressable>

          <Text className="mt-2 font-bodyBold text-head2 text-primary-500">
            {profile?.username || 'Guest User'}
          </Text>
          <Text className="text-sm text-primary-300">
            {isBusy ? 'Loading...' : (email ?? 'No email')}
          </Text>
        </View>

        <View className="mb-10 space-y-4 rounded-2xl bg-background-0 p-6 shadow-md">
          <View className="flex-row justify-between">
            <Text className="font-body text-primary-200">Member since</Text>
            <Text className="font-bodyBold text-primary-500">
              {isBusy ? 'Loading...' : (memberSince ?? 'N/A')}
            </Text>
          </View>
          <View className="flex-row justify-between">
            <Text className="font-body text-primary-200">DOB</Text>
            <Text className="font-bodyBold text-primary-500">
              {profile?.dob ?? 'N/A'}
            </Text>
          </View>
          <View className="flex-row justify-between">
            <Text className="font-body text-primary-200">Gender</Text>
            <Text className="font-bodyBold text-primary-500">
              {profile?.gender ?? 'N/A'}
            </Text>
          </View>

          <View className="h-[1px] bg-primary-200" />
          <View className="flex-row justify-between">
            <Text className="font-body text-primary-200">Status</Text>
            <Text className="font-bodyBold text-green-600">Active</Text>
          </View>
        </View>
        <View className="gap-1">
          <Text className="font-bodySemiBold text-body2 text-primary-300">
            Theme
          </Text>
          <ThemeToggle />
        </View>
      </View>
      <Pressable onPress={handleLogout} className="button">
        <Text className="text-center font-bodyBold text-body2 text-background-0">
          Log out
        </Text>
      </Pressable>
    </ThemedSafeAreaView>
  );
}
