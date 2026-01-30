import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Image,
    SafeAreaView,
    ActivityIndicator,
    Alert, 
    ScrollView,
} from 'react-native';
import { MaterialIcons, Feather, Entypo } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import * as ImagePicker from 'expo-image-picker';

export default function Profile() {
    const router = useRouter();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // Helper to normalize the user data structure
    const normalizeUser = (data) => ({
        // Prioritize common keys, but ensure all data is preserved
        ...data,
        name: data.name || data.first_name || '',
        lastName: data.lastName || data.last_name || '',
        email: data.email || data.username || '',
        profile_image: data.profile_image || null,
    });

    // Function to load profile from storage
    const loadUser = async () => {
        setLoading(true);
        try {
            const storedUserStr = await SecureStore.getItemAsync('userProfile');
            if (storedUserStr) {
                const storedUser = JSON.parse(storedUserStr);
                const normalizedUser = normalizeUser(storedUser);
                setUser(normalizedUser);
            } else {
                // If no profile data exists, fall back to guest/placeholder
                const token = await SecureStore.getItemAsync('userToken');
                if (!token) {
                    router.replace('/login');
                    return;
                }
                setUser(normalizeUser({ name: 'Guest', email: 'guest@example.com' }));
            }
        } catch (error) {
            console.error('Failed to load user profile:', error);
            setUser(normalizeUser({ name: 'Error Loading', email: 'error@example.com' }));
        } finally {
            setLoading(false);
        }
    };

    // Load user on component mount
    useEffect(() => {
        loadUser();
    }, []);

    // Function to handle image picking and profile update
    const updateProfileImage = async () => {
        if (!user) {
            Alert.alert('Error', 'User data not loaded yet.');
            return;
        }

        try {
            const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (permissionResult.status !== 'granted') {
                Alert.alert('Permission Denied', 'Permission to access photos is required!');
                return;
            }

            const pickerResult = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                allowsEditing: true,
                aspect: [1, 1],
                quality: 0.7,
            });

            if (!pickerResult.canceled) {
                const newImageUri = pickerResult.assets[0].uri;

                // 1. Update the local state with the new URI
                const updatedUser = { 
                    ...user, // CRITICAL: Spread the entire current user state
                    profile_image: newImageUri 
                };
                setUser(updatedUser);

                // 2. Save the COMPLETE updated user object back to SecureStore
                // This prevents other profile fields from being lost and ensures the image persists.
                await SecureStore.setItemAsync('userProfile', JSON.stringify(updatedUser));
                
                Alert.alert('Success', 'Profile image updated and saved locally.');
            }
        } catch (error) {
            console.error('Error picking or updating image:', error);
            Alert.alert('Error', 'Failed to update the profile image.');
        }
    };

    // LOGOUT HANDLER (Maintains persistence by ONLY deleting the auth token)
    const handleLogout = async () => {
        await SecureStore.deleteItemAsync('userToken');
        // DO NOT DELETE 'userProfile' here to keep the image path saved locally.
        router.replace('/login');
    };


    if (loading) {
        return (
            <SafeAreaView style={[styles.container, styles.loadingContainer]}>
                <ActivityIndicator size="large" color="#005A9C" />
            </SafeAreaView>
        );
    }

    if (!user) return null;

    const displayName = `${user.first_name} ${user.lastName}`;

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity
                        onPress={() => {
                            if (router.canGoBack()) router.back();
                            else router.replace('/dashboard');
                        }}
                    >
                        <MaterialIcons name="arrow-back-ios" size={24} color="#005A9C" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>My Profile</Text>
                    <View style={{ width: 24 }} />
                </View>

                {/* Profile Section */}
                <View style={styles.profileSection}>
                    <TouchableOpacity 
                        style={styles.profilePicWrapper} 
                        onPress={updateProfileImage}
                    >
                        <Image
                            source={{
                                uri:
                                    user.profile_image
                                        ? user.profile_image
                                        : 'https://cdn-icons-png.flaticon.com/512/149/149071.png',
                            }}
                            style={styles.profilePic}
                        />
                        <TouchableOpacity 
                            style={styles.editIconWrapper} 
                            onPress={updateProfileImage}
                        >
                            <Feather name="edit-3" size={18} color="#fff" />
                        </TouchableOpacity>
                    </TouchableOpacity>
                    <Text style={styles.userName}>{displayName}</Text>
                </View>

                {/* Menu Options */}
                <View style={styles.menuContainer}>
                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => router.push('/edit-profile')}
                    >
                        <MaterialIcons name="person-outline" size={24} color="#005A9C" />
                        <Text style={styles.menuText}>Edit Profile</Text>
                        <Entypo name="chevron-right" size={22} color="#005A9C" />
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => router.push('/privacy-policy')}
                    >
                        <MaterialIcons name="lock-outline" size={24} color="#005A9C" />
                        <Text style={styles.menuText}>Privacy Policy</Text>
                        <Entypo name="chevron-right" size={22} color="#005A9C" />
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={() => router.push('/settings')}
                    >
                        <MaterialIcons name="settings" size={24} color="#005A9C" />
                        <Text style={styles.menuText}>Settings</Text>
                        <Entypo name="chevron-right" size={22} color="#005A9C" />
                    </TouchableOpacity>

                    <TouchableOpacity
                        style={styles.menuItem}
                        onPress={handleLogout}
                    >
                        <MaterialIcons name="logout" size={24} color="#FF5A5A" />
                        <Text style={[styles.menuText, { color: '#FF5A5A' }]}>Logout</Text>
                        <Entypo name="chevron-right" size={22} color="#FF5A5A" />
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#77CDE0',
    },
    loadingContainer: {
        justifyContent: 'center',
        alignItems: 'center',
    },
    scrollContent: {
        paddingVertical: 20,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 15,
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: '700',
        color: '#005A9C',
    },
    profileSection: {
        alignItems: 'center',
        marginVertical: 40,
        padding: 30,
        marginHorizontal: 20,
        backgroundColor: '#C8EAF7',
        borderRadius: 25,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 10,
    },
    profilePicWrapper: {
        position: 'relative',
        marginBottom: 20,
    },
    profilePic: {
        width: 150,
        height: 150,
        borderRadius: 75,
        backgroundColor: '#C8EAF7',
        borderWidth: 4,
        borderColor: '#77CDE0',
    },
    editIconWrapper: {
        position: 'absolute',
        bottom: 5,
        right: 5,
        backgroundColor: '#005A9C',
        borderRadius: 20,
        padding: 8,
        borderWidth: 2,
        borderColor: '#fff',
    },
    userName: {
        fontSize: 28,
        fontWeight: 'bold',
        color: '#333',
        textAlign: 'center',
    },
    menuContainer: {
        marginHorizontal: 20,
        backgroundColor: '#C8EAF7',
        borderRadius: 25,
        paddingVertical: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 10,
    },
    menuItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 18,
        paddingHorizontal: 25,
        borderBottomColor: '#F5F5F5',
        borderBottomWidth: 1,
    },
    menuText: {
        flex: 1,
        marginLeft: 15,
        fontSize: 18,
        color: '#333',
    },
});
