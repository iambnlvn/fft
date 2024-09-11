const std = @import("std");
const PI = std.math.pi;

pub const FFTError = error{ SizeNotEql, SizeNotPow2 };

fn validateInput(comptime F: type, real: []F, imag: []F) FFTError!void {
    if (real.len != imag.len) return FFTError.SizeNotEql;
    if (@popCount(real.len) != 1) return FFTError.SizeNotPow2;
}

pub fn fft(comptime F: type, real: []F, imag: []F) FFTError!void {
    validateInput(F, real, imag) catch |err| return err;

    shuffle(F, real, imag);
    compute(F, real, imag);
}

pub fn ifft(comptime F: type, real: []F, imag: []F) FFTError!void {
    validateInput(F, real, imag) catch |err| return err;

    // conjugate the imaginary part
    for (imag) |*v| v.* = -v.*;

    fft(F, real, imag) catch unreachable;
    // Normalize the result by dividing by the length and conjugating the imaginary part again
    const lenF = @as(F, @floatFromInt(real.len));
    for (real, imag) |*r, *i| {
        r.* /= lenF;
        i.* /= -lenF;
    }
}

fn shuffle(comptime F: type, real: []F, imag: []F) void {
    std.debug.assert(real.len == imag.len);

    const shrAmount = @bitSizeOf(usize) - @ctz(real.len);

    for (real, 0..) |_, i| {
        const j = @bitReverse(i) >> @intCast(shrAmount);

        if (i < j) {
            std.mem.swap(F, &real[i], &real[j]);
            std.mem.swap(F, &imag[i], &imag[j]);
        }
    }
}
fn compute(comptime F: type, real: []F, imag: []F) void {
    std.debug.assert(real.len == imag.len);

    var step: usize = 1;
    while (step < real.len) : (step <<= 1) {
        var group: usize = 0;
        const jump = step << 1;
        const stp: F = @floatFromInt(step);
        const PI_F = @as(F, PI);

        while (group < step) : (group += 1) {
            const angle: F = -PI_F * @as(F, @floatFromInt(group)) / stp;
            const cosAngle = @cos(angle);
            const sinAngle = @sin(angle);
            var pair = group;

            while (pair < real.len) : (pair += jump) {
                const match = pair + step;
                const tempReal = cosAngle * real[match] - sinAngle * imag[match];
                const tempImag = sinAngle * real[match] + cosAngle * imag[match];
                real[match] = real[pair] - tempReal;
                imag[match] = imag[pair] - tempImag;
                real[pair] += tempReal;
                imag[pair] += tempImag;
            }
        }
    }
}

test "FFT with valid input" {
    var real = [8]f32{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    var imag = [8]f32{ -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 };
    try fft(f32, &real, &imag);

    // Expected output after FFT
    const expected_real = [8]f32{ 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0 };
    const expected_imag = [8]f32{ 0.0, 0.0, 0.0, 0.0, -8.0, 0.0, 0.0, 0.0 };

    for (real, 0..) |value, index| {
        try std.testing.expectEqual(value, expected_real[index]);
    }
    for (imag, 0..) |value, index| {
        try std.testing.expectEqual(value, expected_imag[index]);
    }
}

test "IFFT with valid input" {
    var real = [8]f32{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    var imag = [8]f32{ -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 };
    try ifft(f32, &real, &imag);

    const expected_real = [8]f32{ 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000 };
    const expected_imag = [8]f32{ -0.000, -0.000, -0.000, -0.000, -1.000, -0.000, -0.000, -0.000 };

    try std.testing.expectEqual(real, expected_real);
    try std.testing.expectEqual(imag, expected_imag);
}

test "FFT with unequal array sizes" {
    var real = [8]f32{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    var imag = [7]f32{ -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    const result = fft(f32, &real, &imag);
    try std.testing.expectError(FFTError.SizeNotEql, result);
}

test "FFT with non-power-of-2 array size" {
    var real = [7]f32{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 };
    var imag = [7]f32{ -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    const result = fft(f32, &real, &imag);
    try std.testing.expectError(FFTError.SizeNotPow2, result);
}

test "IFFT with unequal array sizes" {
    var real = [8]f32{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    var imag = [7]f32{ -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    const result = ifft(f32, &real, &imag);
    try std.testing.expectError(FFTError.SizeNotEql, result);
}

test "IFFT with non-power-of-2 array size" {
    var real = [7]f32{ 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 };
    var imag = [7]f32{ -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0 };
    const result = ifft(f32, &real, &imag);
    try std.testing.expectError(FFTError.SizeNotPow2, result);
}
