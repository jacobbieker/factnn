from fact.coordinates import horizontal_to_camera

source_x, source_y = horizontal_to_camera(
    zd_pointing=19.542291810859663, az_pointing=-5.496291781416659,
    zd=19.20888251228536, az=353.0000049564592
)

print(source_x)
print(source_y)

source_x, source_y = horizontal_to_camera(
    zd_pointing=19.87240528421488, az_pointing=351.2856016387259,
    zd=19.721139869847146, az=353.0000049564592
)

print(source_x)
print(source_y)

source_x, source_y = horizontal_to_camera(
    zd_pointing=20.050612767834846, az_pointing=-5.7398384200874375,
    zd=19.62977584441447, az=353.0000049564592
)

print(source_x)
print(source_y)